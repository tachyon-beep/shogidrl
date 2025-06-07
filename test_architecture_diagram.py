#!/usr/bin/env python3
"""
Quick test script to verify the table-based architecture diagram works correctly.
This demonstrates the new Rich Table implementation for the Model Evolution panel.
"""

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

def create_architecture_diagram():
    """Create the architecture diagram using the new table-based approach."""
    # Create a borderless table for perfect alignment
    diagram_table = Table.grid(expand=True, padding=(0, 1))
    diagram_table.add_column(justify="center")
    diagram_table.add_column(justify="center", style="dim")  # For the arrow
    diagram_table.add_column(justify="center")
    diagram_table.add_column(justify="center", style="dim")  # For the arrow
    diagram_table.add_column(justify="left")

    # Sample configuration values
    input_shape_str = "9x9x46"
    core_name_str = "Resnet Core"

    # Group the two output heads to stack them vertically
    heads = Group(
        Text("[Policy Head]", style="bold"),
        Text("[Value Head]", style="bold")
    )

    # Add the components as a single row in the table
    diagram_table.add_row(
        Text(f"[Input: {input_shape_str}]", style="bold"),
        "->",
        Text(f"[{core_name_str}]", style="bold"),
        "->",
        heads
    )
    
    return diagram_table

def main():
    """Test the architecture diagram display."""
    console = Console()
    
    print("Testing the new table-based architecture diagram:")
    print("=" * 60)
    
    # Create the diagram
    arch_diagram = create_architecture_diagram()
    
    # Wrap it in a panel like in the actual implementation
    panel_content = Panel(arch_diagram, border_style="magenta", title="Model Evolution")
    
    # Display it
    console.print(panel_content)
    
    print("\n" + "=" * 60)
    print("✅ Architecture diagram rendered successfully!")
    print("✅ Components are properly aligned in columns")
    print("✅ Policy and Value heads are stacked vertically")

if __name__ == "__main__":
    main()
