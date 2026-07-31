{
  description = "Semantic memory search CLI with token-efficient retrieval";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      mkRecall = pkgs:
        let model = pkgs.callPackage ./nix/model.nix { };
        in
        pkgs.rustPlatform.buildRustPackage {
          pname = "recall";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = with pkgs; [ pkg-config makeWrapper ];
          buildInputs = with pkgs; [ openssl sqlite ];

          # Embeddings run in-process, so the binary needs the weights on disk.
          # Without this it falls back to the hf-hub cache and downloads them on
          # first use — a packaged binary must not depend on the network.
          #
          # --set-default, not --set: exporting RECALL_MODEL_PATH by hand still
          # wins, which is how you point a packaged recall at another model.
          postInstall = ''
            wrapProgram $out/bin/recall \
              --set-default RECALL_MODEL_PATH ${model}
          '';

          meta = with pkgs.lib; {
            description = "Semantic memory search CLI with token-efficient retrieval";
            license = licenses.mit;
            mainProgram = "recall";
            platforms = platforms.unix;
          };
        };
    in
    {
      packages = forAllSystems (pkgs: rec {
        recall = mkRecall pkgs;
        default = recall;
      });

      # For consumers that would rather have `pkgs.recall` than reach into
      # `inputs.recall.packages.${system}`.
      overlays.default = final: _prev: {
        recall = mkRecall final;
      };

      # Home Manager module. recall has no config file — every setting is an
      # environment variable, a CLI flag, or a database row — so this module
      # reconciles the database from Nix attributes instead of writing TOML.
      # See nix/home-manager.nix.
      homeModules = rec {
        recall = import ./nix/home-manager.nix { inherit self; };
        default = recall;
      };

      checks = forAllSystems (pkgs: {
        build = mkRecall pkgs;
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            rustc
            cargo
            rust-analyzer
            rustfmt
            clippy
            pkg-config
            openssl
            sqlite
            jq
          ];
        };
      });
    };
}
