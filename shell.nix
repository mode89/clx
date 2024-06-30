{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    rustc
    rust-analyzer
    gdb
    (python3.withPackages (ps: with ps; [
      ipython
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
