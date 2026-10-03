"""Load a deposited checkpoint into the model of the code repository and run it once.

    python load_checkpoint_example.py <checkpoint.pt> [notebook.ipynb]

The model classes live in GAT_LSTM_NN_Publication_superresolution.ipynb, which serves both
stages (a Stage 1 checkpoint has use_query_decoder = False). Only its 22 pure-definition cells
are executed here; they define classes and functions and read no data.
"""
import json, sys, torch
from torch_geometric.data import Data

DEF_CELLS = ["6a5be23c", "66cf027f", "4d3988c6", "a5660cf4", "bcd95251", "28a9e4ff", "srdec01",
             "f2de07a6", "c53b5783", "858f4194", "b18c6667", "8e7ef9e9", "81d71e4b", "9e030261",
             "pubphys01", "7a1e407c", "8f268a8b", "c62bc05c", "d75ee9f2", "srbase01", "pubts01",
             "71b74ab8"]


def load_model(checkpoint, notebook="GAT_LSTM_NN_Publication_superresolution.ipynb", device="cpu"):
    ns = {"__name__": "mooring_definitions"}
    for cell in json.load(open(notebook, encoding="utf-8"))["cells"]:
        if cell.get("id") in DEF_CELLS:
            exec(compile("".join(cell["source"]), f"cell_{cell['id']}", "exec"), ns)
    ck = torch.load(checkpoint, map_location=device, weights_only=True)   # plain dicts and tensors
    c = ck["cfg"]
    model = ns["MooringGATLSTM"](
        n_static_node_features=12, n_dynamic_node_features=22,      # Appendix A feature matrices
        n_static_edge_features=10, n_dynamic_edge_features=4, output_dim=1,
        gat_hidden_dim=c["gat_hidden_dim"], gat_out_dim=c["gat_out_dim"], num_heads=c["num_heads"],
        gat_dropout=c["gat_dropout"], use_global_context=c["use_global_context"],
        lstm_hidden_dim=c["lstm_hidden_dim"], num_lstm_layers=c["num_lstm_layers"],
        lstm_dropout=c["lstm_dropout"], bidirectional=c["lstm_bidirectional"],
        head_hidden_dim=c["head_hidden_dim"], head_dropout=c["head_dropout"],
        use_query_decoder=c.get("use_query_decoder", False),
        query_decoder_heads=c.get("query_decoder_heads", 4),
        query_decoder_ffn_dim=c.get("query_decoder_ffn_dim", 256),
        query_decoder_fourier_bands=c.get("query_decoder_fourier_bands", 8),
        query_decoder_dropout=c.get("query_decoder_dropout", 0.1),
        query_decoder_residual_interp=c.get("query_decoder_residual_interp", True),
        query_decoder_zero_init=c.get("query_decoder_zero_init", True),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.eval(), ck


def chain_graph(n):
    """A chain graph with the shapes of the real inputs (random values: a shape check only)."""
    src = torch.arange(n - 1)
    ei = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], dim=1)   # E = 2(n-1)
    x_raw = torch.randn(n, 12)
    x_raw[:, 0] = torch.linspace(0, 1, n)                      # s / L, normalised arc length
    return Data(x=torch.randn(n, 12), x_raw=x_raw, edge_index=ei, edge_attr=torch.randn(ei.shape[1], 10))


if __name__ == "__main__":
    ckpt = sys.argv[1]
    nb = sys.argv[2] if len(sys.argv) > 2 else "GAT_LSTM_NN_Publication_superresolution.ipynb"
    model, ck = load_model(ckpt, nb)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"epoch {ck['epoch']}  parameters {n_params:,}  query decoder: {model.query_decoder is not None}")
    W, n_in = 100, 4                                          # a 10 s window here; training used W = 1000
    g_in = chain_graph(n_in)
    x_seq = torch.randn(W, n_in, 22)                          # [W, N_in, 22] dynamic node features
    e_seq = torch.randn(W, g_in.edge_index.shape[1], 4)       # [W, E, 4]    dynamic edge features
    q = chain_graph(21) if model.query_decoder is not None else None
    with torch.no_grad():
        y = model(g_in, x_seq, e_seq, q)                      # [W, N_out, 1], standardised tension
    print("output", tuple(y.shape))
