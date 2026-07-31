# Home Manager module for recall.
#
# recall has no config file. Every setting is either an environment variable,
# a CLI flag, or a row in the SQLite database — so a consumer cannot configure
# it by writing a TOML file into `xdg.configFile`. This module exists to hide
# that: you declare collections as Nix attributes and it reconciles the
# database to match on every activation.
{ self }:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.recall;

  # `recall collection list --json` emits the Collection struct verbatim:
  # { id, name, root_path, description, half_life_days, created_at }.
  managedNames = builtins.attrNames cfg.collections;

  shQuote = lib.escapeShellArg;

  # One reconcile block per declared collection. `describe` and `half-life`
  # are set-or-replace and clear when given no value, so passing the option
  # through unconditionally is both idempotent and self-healing: it repairs
  # drift if someone edited the database by hand.
  reconcileOne = name: c: ''
    path=${shQuote c.path}
    if [ ! -d "$path" ]; then
      echo "recall: skipping collection ${name} — path does not exist: $path" >&2
    else
      if ! echo "$existing" | jq -e --arg n ${shQuote name} 'any(.[]; .name == $n)' >/dev/null; then
        echo "recall: registering collection ${name} -> $path"
        $DRY_RUN_CMD recall collection add "$path" --name ${shQuote name}
      fi
      $DRY_RUN_CMD recall collection describe ${shQuote name} ${
        lib.optionalString (c.description != null) (shQuote c.description)
      }
      $DRY_RUN_CMD recall collection half-life ${shQuote name} ${
        lib.optionalString (c.halfLifeDays != null) (toString c.halfLifeDays)
      }
    fi
  '';

  # `index` matches array elements exactly. `inside`/`contains` would not:
  # jq gives those substring semantics for strings, so a collection named
  # "ari" would be treated as managed whenever "aria" is declared.
  pruneBlock = ''
    echo "$existing" \
      | jq -r --argjson keep ${shQuote (builtins.toJSON managedNames)} \
          '.[] | select(.name as $n | ($keep | index($n)) == null) | .name' \
      | while read -r stale; do
          [ -n "$stale" ] || continue
          echo "recall: removing unmanaged collection $stale (not declared in Nix)" >&2
          $DRY_RUN_CMD recall collection remove "$stale"
        done
  '';

  envVars = { RECALL_DB_PATH = cfg.dbPath; };

in
{
  options.programs.recall = {
    enable = lib.mkEnableOption "recall, a semantic search CLI over markdown vaults";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.recall;
      defaultText = lib.literalExpression "recall.packages.\${system}.recall";
      description = "The recall package to install.";
    };

    dbPath = lib.mkOption {
      type = lib.types.str;
      default = "${config.xdg.dataHome}/recall/memory.sqlite";
      defaultText = lib.literalExpression "\"\${config.xdg.dataHome}/recall/memory.sqlite\"";
      description = ''
        Index location, exported as `RECALL_DB_PATH`.

        A schema or chunker change recreates this file from scratch — recall
        carries no migration code by design.
      '';
    };

    collections = lib.mkOption {
      default = { };
      description = ''
        Collections to keep registered. The attribute name is the collection
        name used by `--collection` and by the MCP `collection` argument.

        Reconciled on every activation: missing collections are added,
        descriptions and half-lives are set-or-replaced, and a collection
        whose path does not exist yet is skipped with a warning rather than
        failing the switch.
      '';
      example = lib.literalExpression ''
        {
          aria = {
            path = "''${config.home.homeDirectory}/Obsidian/ARIA";
            description = "ARIA's own memory, daily notes, and skills";
            halfLifeDays = 30;
          };
        }
      '';
      type = lib.types.attrsOf (lib.types.submodule {
        options = {
          path = lib.mkOption {
            type = lib.types.str;
            description = "Root directory this collection indexes.";
          };
          description = lib.mkOption {
            type = lib.types.nullOr lib.types.str;
            default = null;
            description = ''
              Returned alongside this collection's hits, so a reading agent
              knows what the corpus is. Null clears it.
            '';
          };
          halfLifeDays = lib.mkOption {
            type = lib.types.nullOr lib.types.numbers.positive;
            default = null;
            description = ''
              Recency half-life for this collection's chunks.

              Per-collection because corpora age at different rates: a
              half-life tuned for a fast-turnover vault pins a slow archive to
              the decay floor, and one tuned for the archive leaves the fast
              vault undifferentiated. Null falls back to recall's built-in
              default.
            '';
          };
        };
      });
    };

    pruneUnmanagedCollections = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Remove collections present in the database but absent from
        {option}`programs.recall.collections`.

        This makes the Nix config authoritative. It is destructive: removing a
        collection drops its files, chunks, and embeddings, so a mistyped
        attribute name costs a re-index and a re-embed of that corpus.
      '';
    };

    watch = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = ''
          Run `recall watch` as a systemd user service, auto-indexing each
          collection root on change.

          The service indexes once at start (`recall watch` itself only reacts
          to changes, so without this a fresh machine would keep an empty
          index until someone ran `recall index` by hand).
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package ];

    home.sessionVariables = envVars;

    home.activation.recallCollections =
      lib.hm.dag.entryAfter [ "writeBoundary" ] ''
        PATH="${cfg.package}/bin:${pkgs.jq}/bin:$PATH"
        ${lib.concatStringsSep "\n" (lib.mapAttrsToList (n: v: "export ${n}=${shQuote v}") envVars)}

        $DRY_RUN_CMD mkdir -p "$(dirname ${shQuote cfg.dbPath})"

        # No `2>/dev/null || echo '[]'` here. `collection list` already returns
        # `[]` on a database that does not exist yet, so the fallback never
        # covered the fresh-install case it looked like it was for. The only
        # thing it caught was a real error, and it reported that as "nothing is
        # registered" — so every block below took the add branch and
        # `collection add` failed on `UNIQUE constraint failed:
        # collections.name` against rows that were there all along. The
        # activation script runs under `set -eu -o pipefail`, so letting the
        # failure through stops it on recall's own message instead. Same rule
        # the store follows: a database that reports itself as empty is
        # indistinguishable from a fresh install, and every consumer draws the
        # wrong conclusion from that.
        existing="$(recall collection list --json)"

        ${lib.concatStringsSep "\n" (lib.mapAttrsToList reconcileOne cfg.collections)}
        ${lib.optionalString cfg.pruneUnmanagedCollections pruneBlock}
      '';

    systemd.user.services.recall = lib.mkIf cfg.watch.enable {
      Unit = {
        Description = "recall vault watcher";
        After = [ "graphical-session.target" ];
      };
      Service = {
        Type = "simple";
        # `recall watch` only reacts to changes. Index once first so a fresh
        # machine converges without a manual step; subsequent runs skip
        # unchanged files by mtime and cost almost nothing.
        ExecStartPre = "${cfg.package}/bin/recall index";
        ExecStart = "${cfg.package}/bin/recall watch";
        Restart = "on-failure";
        RestartSec = 10;
        Environment = lib.mapAttrsToList (n: v: "${n}=${v}") envVars;
      };
      Install.WantedBy = [ "default.target" ];
    };
  };
}
