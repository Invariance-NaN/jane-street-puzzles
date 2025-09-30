{
  description = "Programmatic solvers for some Jane Street puzzles";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, systems, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
  let
    mkPythonSet = pkgs:
      let
        python = pkgs.python3;
        pythonBase = pkgs.callPackage pyproject-nix.build.packages { inherit python; };
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      in pythonBase.overrideScope (
        pkgs.lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
        ]
      );

    perSystem = fn: nixpkgs.lib.genAttrs (import systems) (system:
    let
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3;
      pythonBase = pkgs.callPackage pyproject-nix.build.packages { inherit python; };
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      pythonSet = pythonBase.overrideScope (
        pkgs.lib.composeManyExtensions [
          pyproject-build-systems.overlays.wheel
          overlay
        ]
      );
    in
    fn {
      inherit pkgs system;
      uv = {
        inherit workspace pythonSet;
        inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
      };
    });
  in
  {
    packages = perSystem ({ uv, ... }: {
      default = uv.mkApplication {
        venv = uv.pythonSet.mkVirtualEnv "jane-street-puzzles-env" uv.workspace.deps.default;
        package = uv.pythonSet.jane-street-puzzles;
      };
    });

    devShells = perSystem ({ pkgs, ... }: {
      default = pkgs.mkShell { packages = [
        pkgs.bashInteractive
        pkgs.uv
      ]; };
    });
  };
}
