{
  description = "Semantic memory search CLI with token-efficient retrieval";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      mkRecall = pkgs: pkgs.rustPlatform.buildRustPackage {
        pname = "recall";
        version = "0.1.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;

        nativeBuildInputs = with pkgs; [ pkg-config ];
        buildInputs = with pkgs; [ openssl sqlite ];

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
