# 🧭 Pathfinding Algorithm Visualizer

This is an interactive **pathfinding algorithm visualizer** built with **Pygame**. It allows you to create a grid, place start and end nodes, and visualize how different algorithms search for the shortest path in real-time.

## 🚀 Features

- Visualize 5 popular pathfinding algorithms:
  - A* (A-Star)
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Dijkstra's Algorithm
  - Uniform Cost Search (UCS)
- Dynamic and colorful grid interaction
- Interactive GUI with animated buttons
- Console output showing path length and steps
- "Impossible" message if no path exists

## 🧠 Algorithms Explained

- **A\*** – Heuristic-based, fast, and optimal.
- **BFS** – Explores all neighbors level by level.
- **DFS** – Dives deep first, less optimal.
- **Dijkstra** – Similar to A\*, but no heuristic.
- **UCS** – Like Dijkstra, but uses total path cost.

## 💻 Installation

Ensure you have Python and Pygame installed. Use the following commands:

```bash
# If you're using Windows:
py -m pip install pygame

# If you're using WSL or Linux:
pip3 install pygame
