## Server

To start the server:

```sh
cd server
nix-shell
uv run src/paraformer.py
```

## Client

To start the client:

```sh
cd client
nix-shell
sudo -E python manual.py
```
