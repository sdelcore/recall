# The embedding model, staged into the Nix store.
#
# recall embeds in-process (see src/embedder.rs), so it needs config.json,
# tokenizer.json and model.safetensors on disk before it can vectorize
# anything. Left alone the binary falls back to the hf-hub cache and downloads
# them on first use, which turns the first `recall embed` into a network
# operation and fails outright on an offline or sandboxed host. Pinning the
# three files here puts them in the closure instead: the wrapper points
# RECALL_MODEL_PATH at this derivation and embedder.rs never reaches the
# network.
#
# The cost is ~134 MB of closure, almost all of it model.safetensors.
#
# The revision is pinned rather than tracking `main`. fetchurl hashes content,
# so a re-tagged `main` would break the build loudly instead of silently
# changing every vector. Bumping it carries the same consequence as changing
# EMBEDDING_MODEL: the stored vectors become incomparable and the index has to
# be rebuilt.
{ lib, fetchurl, runCommand }:

let
  repo = "BAAI/bge-small-en-v1.5";
  rev = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";

  fetchFile = name: hash: fetchurl {
    url = "https://huggingface.co/${repo}/resolve/${rev}/${name}";
    inherit hash;
  };

  files = {
    "config.json" = fetchFile "config.json"
      "sha256-CU+OiRuTLyAAySz8ZjusTGIGn12K9bUnjEMGrvMIR1A=";
    "tokenizer.json" = fetchFile "tokenizer.json"
      "sha256-0kGmDV6PBMwbKz6e96SSGye/Um2fYFCrkPkmeh+eXGY=";
    "model.safetensors" = fetchFile "model.safetensors"
      "sha256-PJ8xZlRHyJEVF2IHYiANIkWiUY1ucgisx4zZ2zF+Ia0=";
  };
in
runCommand "bge-small-en-v1.5" {
  meta = {
    description = "BAAI/bge-small-en-v1.5 sentence embedding model, 384 dimensions";
    homepage = "https://huggingface.co/${repo}";
    license = lib.licenses.mit;
  };
} ''
  mkdir -p $out
  ${lib.concatStringsSep "\n"
    (lib.mapAttrsToList (name: src: "cp ${src} $out/${name}") files)}
''
