{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    rustc
    rust-analyzer
    (python3.withPackages (ps: with ps; [
      mypy
      pylint
      pytest
      pytest-cov
    ]))
  ];
  shellHook = ''
    export PYTHONPATH=$PWD/ext/target
  '';
}
