import torch
import torch.nn as nn
import chess
import random
from typing import Optional
from chess_tournament.players import Player

class RLTransformerPlayer(Player):
    """
    All-in-one Chess Transformer Player. 
    Matches 10k RL Training Logic exactly.
    """
    def __init__(self, name="RL_Transformer", model_path="parom23/chess_transformer"):
        super().__init__(name)
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        # token Ids
        self.piece_token_map = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }

    class ChessModel(nn.Module): 
        def __init__(self):
            super().__init__()
            H = 256
            self.piece_embed = nn.Embedding(13, H)
            self.square_embed = nn.Embedding(64, H)
            layer = nn.TransformerEncoderLayer(
                d_model=H, nhead=8, dim_feedforward=1024, 
                batch_first=True, activation="gelu"
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=8)
            self.policy_head = nn.Sequential(nn.Linear(H*64, 1024), nn.GELU(), nn.Linear(1024, 4096))
            self.value_head = nn.Sequential(nn.Linear(H*64, 512), nn.GELU(), nn.Linear(512, 1), nn.Tanh())

        def forward(self, input_ids):
            batch = input_ids.size(0)
            pos = torch.arange(64, device=input_ids.device).unsqueeze(0).expand(batch, 64)
            x = self.piece_embed(input_ids) + self.square_embed(pos)
            x = self.transformer(x).reshape(batch, -1)
            return {"policy_logits": self.policy_head(x), "value": self.value_head(x)}

    def _board_to_tokens(self, board):
        tokens = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None: tokens.append(0)
            else:
                val = self.piece_token_map.get(piece.piece_type, 0)
                if piece.color == chess.BLACK: val += 6
                tokens.append(val)
        # IMPORTANT: Must be LongTensor (Integers) for Embedding layer
        return torch.tensor(tokens, dtype=torch.long, device=self.device).clamp(0, 12).unsqueeze(0)

    def _load_model(self):
        if self.model is None:
            self.model = self.ChessModel()
            from huggingface_hub import hf_hub_download
            try:
                try:
                    path = hf_hub_download(repo_id=self.model_path, filename="model.safetensors")
                    from safetensors.torch import load_file
                    sd = load_file(path)
                except:
                    path = hf_hub_download(repo_id=self.model_path, filename="pytorch_model.bin")
                    sd = torch.load(path, map_location="cpu")
                
                # Shield: Clean any NaNs from the 10k training run
                for k in sd:
                    if torch.is_tensor(sd[k]):
                        sd[k] = torch.nan_to_num(sd[k], nan=0.0)
                
                self.model.load_state_dict(sd)
                self.model.to(self.device).eval()
                print(f"[{self.name}] Model successfully loaded to {self.device}")
            except Exception as e:
                print(f"[{self.name}] Load Error: {e}")

    def get_move(self, fen: str) -> str:
        try:
            self._load_model()
            board = chess.Board(fen)
            tokens = self._board_to_tokens(board)
            
            with torch.no_grad():
                out = self.model(tokens)
                logits = out["policy_logits"].squeeze(0)

            # Mask illegal moves
            legal_moves = list(board.legal_moves)
            mask = torch.full((4096,), -1e9, device=self.device)
            for m in legal_moves:
                mask[m.from_square * 64 + m.to_square] = 0

            # Probabilistic sampling with temperature to avoid loops
            probs = torch.softmax((logits + mask) / 0.2, dim=0)
            best_idx = torch.multinomial(probs, 1).item()
            
            move = chess.Move(best_idx // 64, best_idx % 64)
            
            # Auto-promotion to Queen
            if board.piece_at(move.from_square) and board.piece_at(move.from_square).piece_type == chess.PAWN:
                if (move.to_square // 8 == 7) or (move.to_square // 8 == 0):
                    move.promotion = chess.QUEEN
            
            return move.uci()
        except Exception:
            # Final fallback to ensure the tournament doesn't stall
            board = chess.Board(fen)
            moves = [m.uci() for m in board.legal_moves]
            return random.choice(moves) if moves else "0000"

