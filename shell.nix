{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    rustc
    rust-analyzer
    (python3.withPackages (ps: with ps; [
      pip
      pylint
      pytest
      pytest-cov
    ]))
  ];
}
