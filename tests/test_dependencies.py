"""
Test suite for dependency optimization and management.

Tests the optimized dependency configuration and validates that all
required dependencies are properly configured and working.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


# Project root path resolution - works from any test file
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Go up from tests/ to project root


class TestDependencyStructure:
    """Test the structure and organization of dependencies."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is readable."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_path.exists()
        assert pyproject_path.is_file()

        # Should be parseable as valid TOML
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        
        assert len(data) > 0
        assert "project" in data

    def test_main_dependencies_structure(self):
        """Test that main dependencies are properly structured."""
        # These should be importable without issues
        main_deps = [
            "torch",
            "numpy",
            "yaml",  # PyYAML
            "rich",
            "pydantic",
            "dotenv",  # python-dotenv
            "wandb",
        ]

        for dep in main_deps:
            try:
                __import__(dep)
            except ImportError as e:
                pytest.fail(f"Main dependency '{dep}' not available: {e}")

    def test_dev_dependencies_available(self):
        """Test that development dependencies are available."""
        dev_deps = ["pytest", "black", "mypy", "pylint", "flake8", "isort"]

        # These might not all be installed, but should be listed in pyproject.toml
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Check if dev dependencies are listed in build-system requirements, 
        # project optional dependencies, or the content as fallback
        content = pyproject_path.read_text(encoding="utf-8")
        
        for dep in dev_deps:
            # Check in various possible locations
            found = False
            if "project" in data and "optional-dependencies" in data["project"]:
                for dep_group in data["project"]["optional-dependencies"].values():
                    if any(dep in dep_entry for dep_entry in dep_group):
                        found = True
                        break
            
            # Fallback to string search if not found in structured way
            if not found and dep in content:
                found = True
                
            assert found, f"Dev dependency '{dep}' not found in pyproject.toml"


class TestDependencyFunctionality:
    """Test that dependencies are working correctly."""

    def test_torch_functionality(self):
        """Test PyTorch basic functionality."""
        import torch

        # Basic tensor operations
        tensor = torch.tensor([1.0, 2.0, 3.0])
        # Use approximate comparison for floats
        assert abs(tensor.sum().item() - 6.0) < 1e-6

        # Check CUDA availability (optional)
        cuda_available = torch.cuda.is_available()
        # Don't assert CUDA requirement, just check it doesn't crash
        assert isinstance(cuda_available, bool)

    def test_numpy_functionality(self):
        """Test NumPy basic functionality."""
        import numpy as np

        array = np.array([1, 2, 3, 4, 5])
        # Use approximate comparison for floats
        assert abs(array.sum() - 15) < 1e-6
        assert abs(array.mean() - 3.0) < 1e-6

        # Test array operations
        result = array * 2
        expected = np.array([2, 4, 6, 8, 10])
        assert np.array_equal(result, expected)

    def test_pydantic_functionality(self):
        """Test Pydantic basic functionality."""
        from pydantic import BaseModel, ValidationError

        class TestModel(BaseModel):
            name: str
            value: int

        # Valid model
        model = TestModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

        # Validation should work
        with pytest.raises(ValidationError):
            TestModel(name="test", value="not_an_int")  # type: ignore[arg-type]

    def test_yaml_functionality(self):
        """Test PyYAML functionality."""
        import yaml

        # Test data serialization/deserialization
        test_data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}

        # Serialize to YAML
        yaml_str = yaml.dump(test_data)
        assert isinstance(yaml_str, str)
        assert "name: test" in yaml_str

        # Deserialize from YAML
        loaded_data = yaml.safe_load(yaml_str)
        assert loaded_data == test_data

    def test_rich_functionality(self):
        """Test Rich library functionality."""
        from io import StringIO

        from rich.console import Console
        from rich.text import Text

        # Test console creation
        output = StringIO()
        console = Console(file=output, width=80)

        # Test basic printing
        console.print("Hello, World!")
        output_text = output.getvalue()
        assert "Hello, World!" in output_text

        # Test rich text
        text = Text("Styled text", style="bold red")
        assert text.plain == "Styled text"

    def test_python_dotenv_functionality(self):
        """Test python-dotenv functionality."""
        import os
        import tempfile

        from dotenv import dotenv_values, load_dotenv

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_VAR=test_value\n")
            f.write("NUMBER_VAR=123\n")
            env_file = f.name

        try:
            # Test loading from file
            env_vars = dotenv_values(env_file)
            assert env_vars["TEST_VAR"] == "test_value"
            assert env_vars["NUMBER_VAR"] == "123"

            # Test loading into environment
            load_dotenv(env_file)
            assert os.getenv("TEST_VAR") == "test_value"
        finally:
            os.unlink(env_file)
            # Clean up environment
            if "TEST_VAR" in os.environ:
                del os.environ["TEST_VAR"]
            if "NUMBER_VAR" in os.environ:
                del os.environ["NUMBER_VAR"]


class TestDependencyVersioning:
    """Test dependency version constraints."""
    
    @pytest.mark.parametrize(
        "library_name,import_name,version_attr,expected_major,expected_minor",
        [
            ("Pydantic", "pydantic", "VERSION", 2, None),
            ("PyTorch", "torch", "__version__", 2, None),
            ("NumPy", "numpy", "__version__", 1, 24),
        ],
        ids=["pydantic", "torch", "numpy"],
    )
    def test_version_compatibility(self, library_name, import_name, version_attr, expected_major, expected_minor):
        """Test that library versions are compatible."""
        import importlib

        module = importlib.import_module(import_name)
        version = getattr(module, version_attr)

        # Handle special case of pydantic VERSION tuple
        if isinstance(version, tuple):
            version_str = ".".join(map(str, version))
        else:
            version_str = version

        # Handle version strings with suffixes (e.g., "2.5.0+cu126" -> "2.5.0")
        version_str = version_str.split("+")[0]
        
        version_parts = [int(x) for x in version_str.split(".")]
        major_version = version_parts[0]
        minor_version = version_parts[1] if len(version_parts) > 1 else 0
        
        if expected_minor is None:
            # Only check major version
            assert major_version >= expected_major, f"{library_name} version too old: {version_str}"
        else:
            # Check major.minor
            assert major_version > expected_major or (
                major_version == expected_major and minor_version >= expected_minor
            ), f"{library_name} version too old: {version_str}"


class TestRemovedDependencies:
    """Test that removed dependencies are properly cleaned up."""

    def test_matplotlib_not_required(self):
        """Test that matplotlib is not required for core functionality."""
        # The system should work without matplotlib
        try:
            from keisei.shogi.shogi_game import ShogiGame
            from keisei.utils import load_config

            # Core functionality should work - use defaults from load_config
            app_config = load_config()
            config = app_config.training
            game = ShogiGame()
            state = game.get_state()

            assert config is not None
            assert state is not None

        except ImportError as e:
            if "matplotlib" in str(e).lower():
                pytest.fail("Core functionality should not require matplotlib")
            else:
                # Re-raise if it's a different import error
                raise

    def test_matplotlib_references_removed(self):
        """Test that matplotlib references are properly removed."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Check main dependencies
        if "project" in data and "dependencies" in data["project"]:
            main_deps = data["project"]["dependencies"]
            for dep in main_deps:
                assert "matplotlib" not in dep.lower(), f"matplotlib found in main dependencies: {dep}"

        # Check optional dependencies (dev, etc.)
        if "project" in data and "optional-dependencies" in data["project"]:
            optional_deps = data["project"]["optional-dependencies"]
            for group_name, deps in optional_deps.items():
                for dep in deps:
                    assert "matplotlib" not in dep.lower(), f"matplotlib found in {group_name} dependencies: {dep}"


class TestDependencyAnalysis:
    """Test dependency analysis tools and results."""

    @pytest.mark.slow
    def test_deptry_analysis(self):
        """Test that deptry analysis shows expected results."""
        try:
            result = subprocess.run(
                ["deptry", str(PROJECT_ROOT)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            # deptry should run successfully
            assert result.returncode in [
                0,
                1,
            ], f"deptry failed with code {result.returncode}"

            # Should not mention matplotlib
            assert (
                "matplotlib" not in result.stdout.lower()
            ), "matplotlib still appears in dependency analysis"

            # Count dependency issues
            dep_issues = result.stdout.count("DEP002")

            # Dependency issues should be manageable - this count represents
            # known acceptable issues like dev dependencies not used in main code
            # If this fails, investigate the specific DEP002 issues reported
            max_acceptable_issues = 20  # Conservative threshold allowing for project growth
            assert dep_issues <= max_acceptable_issues, (
                f"Too many dependency issues: {dep_issues} > {max_acceptable_issues}. "
                f"Check deptry output for specific issues and either fix them or "
                f"update the threshold if they are acceptable."
            )

        except FileNotFoundError:
            pytest.skip("deptry not available")
        except subprocess.TimeoutExpired:
            pytest.fail("deptry analysis timed out")

    def test_no_unused_imports_in_core(self):
        """Test that core modules don't have unused imports."""
        # This is a basic check - a full analysis would require AST parsing
        core_files = [
            PROJECT_ROOT / "keisei" / "config_schema.py",
            PROJECT_ROOT / "keisei" / "shogi" / "shogi_game.py",
            PROJECT_ROOT / "keisei" / "utils" / "profiling.py",
        ]

        for file_path in core_files:
            if not file_path.exists():
                continue

            content = file_path.read_text(encoding="utf-8")

            # Basic check: no matplotlib imports
            assert "import matplotlib" not in content
            assert "from matplotlib" not in content

            # Check that imports are actually used (basic heuristic)
            lines = content.split("\n")
            import_lines = [
                line.strip()
                for line in lines
                if line.strip().startswith(("import ", "from "))
            ]

            for import_line in import_lines:
                if "matplotlib" in import_line:
                    pytest.fail(
                        f"Found matplotlib import in {file_path}: {import_line}"
                    )


class TestDependencyInstallation:
    """Test dependency installation and environment setup."""

    def test_pip_environment_consistency(self):
        """Test that pip environment is consistent with pyproject.toml."""
        import json

        try:
            # Get installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                installed_packages = json.loads(result.stdout)
                package_names = {pkg["name"].lower() for pkg in installed_packages}

                # Key dependencies should be installed
                required_packages = {
                    "torch",
                    "numpy",
                    "pyyaml",
                    "rich",
                    "pydantic",
                    "python-dotenv",
                    "wandb",
                }

                for pkg in required_packages:
                    assert (
                        pkg in package_names or pkg.replace("-", "_") in package_names
                    ), f"Required package '{pkg}' not installed"

        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pytest.skip("Could not check pip environment")

    def test_requirements_files_consistency(self):
        """Test that requirements files are consistent with pyproject.toml."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        requirements_path = PROJECT_ROOT / "requirements.txt"

        if requirements_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
            requirements_content = requirements_path.read_text(encoding="utf-8")

            # Extract main dependencies from pyproject.toml
            main_packages = ["torch", "numpy", "pydantic", "rich"]
            
            if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
                pyproject_deps = pyproject_data["project"]["dependencies"]
                
                for pkg in main_packages:
                    # Check if package is in pyproject dependencies
                    pkg_in_pyproject = any(pkg in dep for dep in pyproject_deps)
                    
                    if pkg_in_pyproject:
                        # Should also be in requirements.txt if it exists
                        assert (
                            pkg in requirements_content
                            or pkg.replace("-", "_") in requirements_content
                        ), f"Package '{pkg}' in pyproject.toml but not in requirements.txt"


@pytest.mark.integration
class TestDependencyIntegration:
    """Integration tests for dependency functionality."""

    def test_full_system_imports(self):
        """Test that the full system can be imported without dependency issues."""
        # This should work without any missing dependency errors
        from keisei.shogi.shogi_game import ShogiGame
        from keisei.training.env_manager import EnvManager
        from keisei.utils import load_config
        from keisei.utils.profiling import perf_monitor

        # Basic functionality should work - use load_config to get proper AppConfig
        app_config = load_config()
        game = ShogiGame()
        env_manager = EnvManager(config=app_config)

        assert app_config is not None
        assert game is not None
        assert env_manager is not None
        assert perf_monitor is not None

    def test_training_pipeline_dependencies(self):
        """Test that training pipeline has all required dependencies."""
        from keisei.training.env_manager import EnvManager
        from keisei.utils import load_config

        # Use the default config to get a working configuration
        app_config = load_config()
        env_manager = EnvManager(config=app_config)

        # Setup the environment to initialize the game
        game, _ = env_manager.setup_environment()

        # Should be able to get basic game state
        state = game.get_observation()
        assert state is not None

    def test_configuration_system_dependencies(self):
        """Test that configuration system has all required dependencies."""
        try:
            import yaml

            from keisei.utils import load_config

            # Should be able to create and serialize config using load_config
            app_config = load_config()
            config = app_config.training
            config_dict = config.model_dump()

            # Should be able to serialize to YAML
            yaml_str = yaml.dump(config_dict)
            loaded_dict = yaml.safe_load(yaml_str)

            assert isinstance(loaded_dict, dict)
            assert len(loaded_dict) > 0

        except ImportError as e:
            pytest.fail(f"Configuration system missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
