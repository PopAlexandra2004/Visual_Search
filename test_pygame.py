import pygame
from queue import PriorityQueue, Queue

# Initialize Pygame
pygame.init()

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Pathfinding Algorithm Visualizer")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
TURQUOISE = (64, 224, 208)
PURPLE = (128, 0, 128)
GREY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BUTTON_COLOR = (100, 149, 237)
BUTTON_HOVER_COLOR = (70, 130, 180)
SHADOW_COLOR = (30, 30, 30)

# Font for the selection window
FONT = pygame.font.Font(None, 40)
TITLE_FONT = pygame.font.SysFont('Arial', 60)


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # Down
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # Up
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # Right
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # Left
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

# Heuristic function for A*
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# Function to clear previous algorithm's path and exploration
def clear_path(grid):
    for row in grid:
        for spot in row:
            if not spot.is_start() and not spot.is_end() and not spot.is_barrier():
                spot.reset()

# A* algorithm
def algorithm_a_star(draw, grid, start, end):
    clear_path(grid)  # Clear previous paths and explorations
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    show_message("Impossible")
    return False

# BFS algorithm
def algorithm_bfs(draw, grid, start, end):
    clear_path(grid)  # Clear previous paths and explorations
    queue = Queue()
    queue.put(start)
    came_from = {}
    visited = {start}

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current = queue.get()

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.put(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    show_message("Impossible")
    return False

# DFS algorithm
def algorithm_dfs(draw, grid, start, end):
    clear_path(grid)  # Clear previous paths and explorations
    stack = [start]
    came_from = {}
    visited = {start}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current = stack.pop()

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    show_message("Impossible")
    return False

# Dijkstra's algorithm
def algorithm_dijkstra(draw, grid, start, end):
    clear_path(grid)  # Clear previous paths and explorations
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    open_set.put((g_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    show_message("Impossible")
    return False

# UCS algorithm
def algorithm_ucs(draw, grid, start, end):
    clear_path(grid)  # Clear previous paths and explorations
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    open_set.put((g_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    show_message("Impossible")
    return False

def reconstruct_path(came_from, current, draw):
    path = []  # List to store the path coordinates
    while current in came_from:
        path.append(current.get_pos())  # Collect each coordinate in the path
        current = came_from[current]
        current.make_path()
        draw()
    
    path.reverse()  # Reverse the path to start from the beginning
    
    # Print the path to the console in the specified format
    print("Path")
    print("Start:")
    for pos in path:
        print(f"  {pos},")
    print("Goal")
    print(f"Length: {len(path)} steps")

def show_message(message):
    font = pygame.font.SysFont('Arial', 50)
    text_surface = font.render(message, True, (255, 0, 0))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, WIDTH // 2))
    
    WIN.blit(text_surface, text_rect)
    pygame.display.update()
    pygame.time.delay(2000)


# Selection window for choosing algorithm
def draw_gradient_background(win, top_color, bottom_color):
    height = win.get_height()
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        pygame.draw.line(win, (r, g, b), (0, y), (win.get_width(), y))

# Function to draw icons on the buttons
def draw_button_icon(win, algorithm, rect):
    center_x = rect.centerx
    center_y = rect.centery + 20  # Adjust as needed
    if algorithm == "BFS":
        # Draw a simple network icon
        pygame.draw.circle(win, WHITE, (center_x, center_y), 10)
        pygame.draw.line(win, WHITE, (center_x, center_y), (center_x - 20, center_y + 20), 2)
        pygame.draw.line(win, WHITE, (center_x, center_y), (center_x + 20, center_y + 20), 2)
        pygame.draw.circle(win, WHITE, (center_x - 20, center_y + 20), 5)
        pygame.draw.circle(win, WHITE, (center_x + 20, center_y + 20), 5)
    elif algorithm == "A*":
        # Draw a star
        pygame.draw.polygon(win, WHITE, [
            (center_x, center_y - 15),
            (center_x + 5, center_y - 5),
            (center_x + 15, center_y - 5),
            (center_x + 8, center_y + 2),
            (center_x + 10, center_y + 12),
            (center_x, center_y + 7),
            (center_x - 10, center_y + 12),
            (center_x - 8, center_y + 2),
            (center_x - 15, center_y - 5),
            (center_x - 5, center_y - 5)
        ])
    elif algorithm == "DFS":
        # Draw a path icon
        pygame.draw.lines(win, WHITE, False, [
            (center_x - 15, center_y + 15),
            (center_x - 10, center_y - 10),
            (center_x + 5, center_y + 5),
            (center_x + 15, center_y - 15)
        ], 2)
    elif algorithm == "Dijkstra":
        # Draw a grid with a highlighted path
        pygame.draw.rect(win, WHITE, (center_x - 15, center_y - 15, 30, 30), 2)
        pygame.draw.line(win, WHITE, (center_x - 15, center_y - 15), (center_x + 15, center_y + 15), 2)
    elif algorithm == "UCS":
        # Draw a clock icon
        pygame.draw.circle(win, WHITE, (center_x, center_y), 12, 2)
        pygame.draw.line(win, WHITE, (center_x, center_y), (center_x, center_y - 8), 2)
        pygame.draw.line(win, WHITE, (center_x, center_y), (center_x + 5, center_y), 2)

# Function to draw a rounded button with hover effect and icon
def draw_button(win, rect, text, is_hovered):
    color = BUTTON_HOVER_COLOR if is_hovered else BUTTON_COLOR
    shadow_rect = rect.copy()
    shadow_rect.x += 5
    shadow_rect.y += 5
    pygame.draw.rect(win, SHADOW_COLOR, shadow_rect, border_radius=20)
    pygame.draw.rect(win, color, rect, border_radius=20)
    text_surface = FONT.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=(rect.centerx, rect.centery - 20))
    win.blit(text_surface, text_rect)
    draw_button_icon(win, text, rect)

# Updated algorithm selection window
def algorithm_selection_window():
    # Gradient colors
    top_color = (0, 102, 204)
    bottom_color = (0, 0, 51)
    # Define button rectangles
    buttons = {
        "BFS": pygame.Rect(75, 350, 125, 150),
        "A*": pygame.Rect(225, 350, 125, 150),
        "DFS": pygame.Rect(375, 350, 125, 150),
        "Dijkstra": pygame.Rect(525, 350, 125, 150),
        "UCS": pygame.Rect(675, 350, 125, 150),
    }
    title_surface = TITLE_FONT.render("Choose an Algorithm", True, WHITE)
    clock = pygame.time.Clock()

    # Main loop for selection window
    while True:
        clock.tick(60)
        draw_gradient_background(WIN, top_color, bottom_color)
        WIN.blit(title_surface, (WIDTH // 2 - title_surface.get_width() // 2, 100))

        # Draw each button with hover effect
        for algorithm, rect in buttons.items():
            is_hovered = rect.collidepoint(pygame.mouse.get_pos())
            # Scale the button slightly when hovered
            if is_hovered:
                scaled_rect = rect.inflate(10, 10)
            else:
                scaled_rect = rect
            draw_button(WIN, scaled_rect, algorithm, is_hovered)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for algorithm, rect in buttons.items():
                    if rect.collidepoint(event.pos):
                        return {
                            "BFS": algorithm_bfs,
                            "A*": algorithm_a_star,
                            "DFS": algorithm_dfs,
                            "Dijkstra": algorithm_dijkstra,
                            "UCS": algorithm_ucs,
                        }[algorithm]

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def main(win, width):
    # (Function remains unchanged)
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # Left mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row >= ROWS or col >= ROWS:
                    continue
                spot = grid[row][col]

                if not start and spot != end:
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # Right mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row >= ROWS or col >= ROWS:
                    continue
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    algorithm = algorithm_selection_window()
                    if algorithm:
                        algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()
if __name__ == "__main__":
    main(WIN, WIDTH)