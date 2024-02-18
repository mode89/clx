{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    rustc
    (writeShellScriptBin "rust-analyzer" ''
      export CARGO_TARGET_DIR=.rust-analyzer
      exec ${rust-analyzer}/bin/rust-analyzer "$@"
    '')
    (python3.withPackages (ps: with ps; [
      pip
      pylint
      pytest
      pytest-cov
    ]))
  ];
}
