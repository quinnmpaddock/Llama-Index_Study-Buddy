{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs { inherit system; };
          }
        );

      /*
        Change this value ({major}.{min}) to
        update the Python virtual-environment
        version. When you do this, make sure
        to delete the `.venv` directory to
        have the hook rebuild it for the new
        version, since it won't overwrite an
        existing one. After this, reload the
        development shell to rebuild it.
        You'll see a warning asking you to
        do this when version mismatches are
        present. For safety, removal should
        be a manual step, even if trivial.
      */
      version = "3.12";
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        let
          concatMajorMinor =
            v:
            pkgs.lib.pipe v [
              pkgs.lib.versions.splitVersion
              (pkgs.lib.sublist 0 2)
              pkgs.lib.concatStrings
            ];

          python = pkgs."python${concatMajorMinor version}";

          linuxDisplayLibs = with pkgs; [
            libGL
            libxkbcommon
            xorg.libX11
            xorg.libxcb
            glib
            fontconfig
            stdenv.cc.cc.lib
          ];

          # numpy-pinned = python.pkgs.numpy.overridePythonAttrs (oldAttrs: rec {
          #   version = "1.26.4";
          #   src = pkgs.fetchPypi {
          #     pname = "numpy";
          #     inherit version;
          #     # Update hash for each new pkg version.
          #     # Run 'nix-shell -p nix-prefetch-github' then 'nix-prefetch-url' or just
          #     # use the dummy hash below and Nix will tell you the correct one.
          #     hash = "sha256-7M69S9L0S1mJmD/W8mXmZ3S0zX0zX0zX0zX0zX0zX0z=";
          #   };
          #   # Disable tests to speed up the shell entry
          #   doCheck = false;
          # });
        in
        {
          default = pkgs.mkShell {
            venvDir = ".venv";

            buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux linuxDisplayLibs;

            postShellHook = ''
              venvVersionWarn() {
              	local venvVersion
              	venvVersion="$("$venvDir/bin/python" -c 'import platform; print(platform.python_version())')"

              	[[ "$venvVersion" == "${python.version}" ]] && return

              	cat <<EOF
              Warning: Python version mismatch: [$venvVersion (venv)] != [${python.version}]
                       Delete '$venvDir' and reload to rebuild for version ${python.version}
              EOF
              }
              venvVersionWarn

              if [ -f .env ]; then
                # Safe way to load key-value pairs without executing arbitrary code
                export $(grep -v '^#' .env | xargs)
              else
                echo "Note: No .env file found. Create one to load secrets like GROQ_API_KEY."
              fi
               
            '';

            packages = with python.pkgs; [
              venvShellHook
              pip

              # Add whatever else you'd like here.
              pkgs.basedpyright

              # pkgs.black
              # or
              # python.pkgs.black

              pkgs.ruff
              # or
              # python.pkgs.ruff
              numpy_1
              pandas
              setuptools
              wheel

            ];
          };
        }
      );
    };
}
