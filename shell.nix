{ pkgs ? import <nixpkgs> {} }:

let
  python38 = let
      pkgs = import (fetchTarball
        "https://github.com/NixOS/nixpkgs/archive/23.05.tar.gz") {};
    in pkgs.python38;
in pkgs.mkShell {
  packages = with pkgs; [
    cargo
    rustc
    rust-analyzer
    gdb
    ncurses
    less
    which
    (python38.withPackages (ps: with ps; [
      ipython
      pylint
      pytest
    ]))
  ];
  shellHook = ''
    export PYTHONPATH=$PWD/ext/target
  '';
}
