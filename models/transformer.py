import torch
import torch.nn as nn
from models.embed import DataEmbedding
from models.enc_dec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)
from models.attn import multiheadattention

class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, enc_in, dec_in, dec_out, pred_len, d_model=512,
                 n_heads=8, enc_layers=3, dec_layers=2, d_ff=512,
                 dropout=0.1, activation='gelu', output_attention=True):
        super(Transformer, self).__init__()

        self.enc_embedding = DataEmbedding(enc_in, d_model)
        self.dec_embedding = DataEmbedding(dec_in, d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    multiheadattention(d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
                for l in range(enc_layers)
            ],
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    multiheadattention(d_model, n_heads),
                    multiheadattention(d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
                for l in range(dec_layers)
            ],
            projection=nn.Linear(d_model, dec_out),
        )
        self.pred_len = pred_len
        self.output_attention = output_attention

    def _make_trg_mask(self, trg):
        B, L, _= trg.shape
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            trg_mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, diagonal=1))
    
        return trg_mask

    def forward(
        self,
        x_enc,
        x_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]


if __name__ == '__main__':
    model = Transformer(2, 2, 2, 24)
    enc_in = torch.randn(32, 48, 2)
    dec_in = torch.randn(32, 48, 2)
    y, attn = model(enc_in, dec_in)
    print(y.shape)