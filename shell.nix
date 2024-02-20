{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      ipython
      mypy
      pylint
      pytest
      pytest-cov
    ]))
  ];
}
