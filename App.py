import pygame
import heapq
import random
from collections import deque # Optimized queue for BFS

# --- CONFIGURATION ---
TILE_SIZE = 25
COLS, ROWS = 34, 26 
MAZE_WIDTH, MAZE_HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE
UI_HEIGHT = 80 # Increased for clearer text
WIDTH, HEIGHT = MAZE_WIDTH, MAZE_HEIGHT + UI_HEIGHT

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 100, 100)   # BFS/Dijkstra Visited
CYAN = (100, 100, 100)  # A* Visited (Dark Grey)
BLUE = (0, 0, 255)      # The "Frontier" (Active Open Set)
GREEN = (0, 255, 0)     # Solution Path
LIGHT_BROWN = (210, 180, 140) # Weighted Terrain (Low cost)
DARK_BROWN = (139, 69, 19)    # Weighted Terrain (High cost)
PURPLE = (40, 40, 40)   # Background
YELLOW = (255, 255, 0)  # Start Node
MAGENTA = (255, 0, 255) # End Node

class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False
        self.parent = None
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.weight = 1

    def draw(self, screen, color=BLACK):
        x, y = self.x * TILE_SIZE, self.y * TILE_SIZE + UI_HEIGHT
        thickness = 2

        # Draw Weights (Terrain)
        if self.weight > 1:
            weight_color = LIGHT_BROWN if self.weight < 10 else DARK_BROWN
            pygame.draw.rect(screen, weight_color, (x + 1, y + 1, TILE_SIZE - 2, TILE_SIZE - 2))

        # Draw Walls
        if self.walls['top']: pygame.draw.line(screen, color, (x, y), (x + TILE_SIZE, y), thickness)
        if self.walls['right']: pygame.draw.line(screen, color, (x + TILE_SIZE, y), (x + TILE_SIZE, y + TILE_SIZE), thickness)
        if self.walls['bottom']: pygame.draw.line(screen, color, (x + TILE_SIZE, y + TILE_SIZE), (x, y + TILE_SIZE), thickness)
        if self.walls['left']: pygame.draw.line(screen, color, (x, y + TILE_SIZE), (x, y), thickness)

    def highlight(self, screen, color):
        x, y = self.x * TILE_SIZE, self.y * TILE_SIZE + UI_HEIGHT
        # Draw slightly smaller to keep walls visible
        pygame.draw.rect(screen, color, (x + 4, y + 4, TILE_SIZE - 8, TILE_SIZE - 8))

    def check_neighbors(self, grid):
        neighbors = []
        x, y = self.x, self.y
        if y > 0 and not grid[x][y - 1].visited: neighbors.append(grid[x][y - 1])
        if x < COLS - 1 and not grid[x + 1][y].visited: neighbors.append(grid[x + 1][y])
        if y < ROWS - 1 and not grid[x][y + 1].visited: neighbors.append(grid[x][y + 1])
        if x > 0 and not grid[x - 1][y].visited: neighbors.append(grid[x - 1][y])
        return random.choice(neighbors) if neighbors else None

    def get_all_neighbors(self, grid):
        neighbors = []
        x, y = self.x, self.y
        if y > 0: neighbors.append(grid[x][y - 1])
        if x < COLS - 1: neighbors.append(grid[x + 1][y])
        if y < ROWS - 1: neighbors.append(grid[x][y + 1])
        if x > 0: neighbors.append(grid[x - 1][y])
        return neighbors

    def get_valid_moves(self, grid):
        moves = []
        x, y = self.x, self.y
        if not self.walls['top']: moves.append(grid[x][y - 1])
        if not self.walls['right']: moves.append(grid[x + 1][y])
        if not self.walls['bottom']: moves.append(grid[x][y + 1])
        if not self.walls['left']: moves.append(grid[x - 1][y])
        return moves

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    # Weighted Manhattan Distance (1.5x) to make A* "Greedy" and fast
    return (abs(a.x - b.x) + abs(a.y - b.y)) * 1.5

def draw_path(end_cell, screen):
    current = end_cell
    while current.parent:
        x = current.x * TILE_SIZE + TILE_SIZE // 2
        y = current.y * TILE_SIZE + TILE_SIZE // 2 + UI_HEIGHT
        px = current.parent.x * TILE_SIZE + TILE_SIZE // 2
        py = current.parent.y * TILE_SIZE + TILE_SIZE // 2 + UI_HEIGHT
        pygame.draw.line(screen, GREEN, (x, y), (px, py), 4)
        current = current.parent

def remove_walls(current, next):
    dx = current.x - next.x
    dy = current.y - next.y
    if dx == 1:
        current.walls['left'] = False
        next.walls['right'] = False
    elif dx == -1:
        current.walls['right'] = False
        next.walls['left'] = False
    if dy == 1:
        current.walls['top'] = False
        next.walls['bottom'] = False
    elif dy == -1:
        current.walls['bottom'] = False
        next.walls['top'] = False

def reset_solver_state(state):
    # Clean up the grid for a new run (keep walls/weights, but wipe memory)
    for row in state['grid']:
        for cell in row:
            cell.parent = None
            cell.g = float('inf')
            cell.h = 0
            cell.f = float('inf')
    
    state['open_set'].clear()
    state['closed_set'].clear()
    state['bfs_queue_list'].clear() # Clears the list (for Dijkstra)
    state['bfs_queue_deque'].clear() # Clears the deque (for BFS)
    state['bfs_visited'].clear()
    state['cells_checked'] = 0

def generate_maze_instant(state):
    # 1. Generate the Maze Structure
    if state['generation_algorithm'] == "RECURSIVE_BACKTRACKING":
        stack = [state['grid'][0][0]]
        stack[0].visited = True
        while stack:
            current_cell = stack[-1]
            next_cell = current_cell.check_neighbors(state['grid'])
            if next_cell:
                next_cell.visited = True
                stack.append(next_cell)
                remove_walls(current_cell, next_cell)
            else:
                stack.pop()
    
    elif state['generation_algorithm'] == "PRIMS":
        start_x, start_y = random.randint(0, COLS-1), random.randint(0, ROWS-1)
        start_cell = state['grid'][start_x][start_y]
        start_cell.visited = True
        frontier = start_cell.get_all_neighbors(state['grid'])
        while frontier:
            current_cell = random.choice(frontier)
            frontier.remove(current_cell)
            maze_neighbors = [n for n in current_cell.get_all_neighbors(state['grid']) if n.visited]
            if maze_neighbors:
                connection = random.choice(maze_neighbors)
                remove_walls(current_cell, connection)
                current_cell.visited = True
                for neighbor in current_cell.get_all_neighbors(state['grid']):
                    if not neighbor.visited and neighbor not in frontier:
                        frontier.append(neighbor)
    
    # 2. Braid the maze (Remove dead ends to allow loops)
    # This makes A* much smarter than BFS
    for _ in range(int(COLS * ROWS * 0.4)): # Remove 40% of dead ends
        rx, ry = random.randint(0, COLS-1), random.randint(0, ROWS-1)
        cell = state['grid'][rx][ry]
        neighbors = cell.get_all_neighbors(state['grid'])
        if neighbors:
            target = random.choice(neighbors)
            remove_walls(cell, target)

    # 3. Apply Terrain Weights
    assign_weights(state)
    state['mode'] = "IDLE"

def assign_weights(state):
    for row in state['grid']:
        for cell in row:
            cell.weight = 1 # Reset all

    if state['costs_enabled']:
        # Create a high-cost "River"
        river_x = COLS // 2
        for y in range(ROWS):
            state['grid'][river_x][y].weight = 50 
            if river_x < COLS-1: state['grid'][river_x+1][y].weight = 50

        # Sprinkle Random Mud Pits
        for _ in range(60):
            rx, ry = random.randint(0, COLS-1), random.randint(0, ROWS-1)
            state['grid'][rx][ry].weight = 20

def init_game_state():
    new_grid = [[Cell(x, y) for y in range(ROWS)] for x in range(COLS)]
    return {
        'grid': new_grid,
        'open_set': [],     # For A*
        'closed_set': set(),# For A*
        'bfs_queue_list': [], # For Dijkstra (Heapq needs a list)
        'bfs_queue_deque': deque(), # For BFS (Deque is O(1))
        'frontier': None,   # Unified queue for BFS/Dijkstra
        'bfs_visited': set(),
        'mode': "IDLE",
        'start_node': None,
        'end_node': None,
        'cells_checked': 0,
        'generation_algorithm': "RECURSIVE_BACKTRACKING",
        'costs_enabled': True,
    }

def reset_maze_grid(state):
    current_algo = state['generation_algorithm']
    # Preserve the costs_enabled setting across resets
    costs_on = state['costs_enabled']
    new_state = init_game_state()
    new_state['generation_algorithm'] = current_algo
    new_state['costs_enabled'] = costs_on
    return new_state

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Algorithm Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18, bold=True) # Clearer font

    # Initialize everything using the function
    state = init_game_state()
    generate_maze_instant(state)

    run = True
    while run:
        screen.fill(WHITE) # Fill background for UI area

        for event in pygame.event.get():
            if event.type == pygame.QUIT: run = False
            
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_r:
                    state = reset_maze_grid(state)
                    generate_maze_instant(state)

                if event.key == pygame.K_1 and state['mode'] == "IDLE":
                    state = init_game_state()
                    state['generation_algorithm'] = "RECURSIVE_BACKTRACKING"
                    generate_maze_instant(state)
                if event.key == pygame.K_2 and state['mode'] == "IDLE":
                    state = init_game_state()
                    state['generation_algorithm'] = "PRIMS"
                    generate_maze_instant(state)

                if event.key == pygame.K_d:
                    state['costs_enabled'] = not state['costs_enabled']
                    assign_weights(state)

                # --- BFS ---
                if event.key == pygame.K_SPACE and state['mode'] in ["IDLE", "DONE"] and state['start_node'] and state['end_node']:
                    reset_solver_state(state) # Clear previous solver's data
                    state['mode'] = "SOLVING_BFS"
                    start = state['start_node']
                    start.g = 0 # Initialize start node cost to 0
                    
                    # Check if weights are enabled to decide queue type
                    # This is the core of the BFS/Dijkstra unification
                    if state['costs_enabled']:
                         heapq.heappush(state['bfs_queue_list'], (0, start))
                         state['frontier'] = [(0, start)] # Use a list for heapq (Dijkstra)
                    else:
                         state['bfs_queue_deque'].append(start)
                         state['frontier'] = deque([start]) # Use a deque for O(1) pops (BFS)
                    
                    state['bfs_visited'].add(start)
                
                # --- A* ---
                if event.key == pygame.K_a and state['mode'] in ["IDLE", "DONE"] and state['start_node'] and state['end_node']:
                    reset_solver_state(state) # Clear previous solver's data
                    state['mode'] = "SOLVING_ASTAR"
                    start = state['start_node']
                    
                    start.g = 0
                    start.h = heuristic(start, state['end_node'])
                    start.f = start.g + start.h
                    heapq.heappush(state['open_set'], (start.f, start))

        if state['mode'] == "IDLE":
            if pygame.mouse.get_pressed()[0]: # Left Click
                mx, my = pygame.mouse.get_pos()
                if my > UI_HEIGHT:
                    grid_x, grid_y = mx // TILE_SIZE, (my - UI_HEIGHT) // TILE_SIZE
                    if 0 <= grid_x < COLS and 0 <= grid_y < ROWS:
                        state['start_node'] = state['grid'][grid_x][grid_y]
            elif pygame.mouse.get_pressed()[2]: # Right Click
                mx, my = pygame.mouse.get_pos()
                if my > UI_HEIGHT:
                    grid_x, grid_y = mx // TILE_SIZE, (my - UI_HEIGHT) // TILE_SIZE
                    if 0 <= grid_x < COLS and 0 <= grid_y < ROWS:
                        state['end_node'] = state['grid'][grid_x][grid_y]

        # Draw maze background
        pygame.draw.rect(screen, PURPLE, (0, UI_HEIGHT, MAZE_WIDTH, MAZE_HEIGHT))

        # --- Create sets for fast drawing lookups (Performance Boost) ---
        open_set_cells = set()
        if state['mode'] == "SOLVING_ASTAR":
            open_set_cells = {cell for f, cell in state['open_set']}
        elif state['mode'] == "SOLVING_BFS":
            if state['costs_enabled']:
                open_set_cells = {cell for g, cell in state['bfs_queue_list']}
                open_set_cells = {cell for g, cell in state['frontier']}
            else:
                open_set_cells = set(state['bfs_queue_deque'])
                open_set_cells = set(state['frontier'])

        # --- MASTER DRAWING LOOP ---
        for x in range(COLS):
            for y in range(ROWS):
                cell = state['grid'][x][y]
                cell.draw(screen, WHITE)
                # Draw solver history
                if cell in state['closed_set']: cell.highlight(screen, CYAN)
                if cell in state['bfs_visited']: cell.highlight(screen, RED)
                
                # Draw Frontier (Blue)
                if cell in open_set_cells:
                    cell.highlight(screen, BLUE)

        # Draw Start and End Points on top of history
        if state['start_node']:
            state['start_node'].highlight(screen, YELLOW)
        if state['end_node']:
            state['end_node'].highlight(screen, MAGENTA)

        # --- LOGIC UPDATES ---
        if state['mode'] == "SOLVING_BFS":
            for _ in range(15):
                if not state['frontier']: state['mode'] = "DONE"; break
                
                # Unified pop operation
                if state['costs_enabled']:
                    if not state['bfs_queue_list']: state['mode'] = "DONE"; break
                    current_g, current = heapq.heappop(state['bfs_queue_list'])
                    # Dijkstra's: Pop the cheapest item from the heap
                    current_g, current = heapq.heappop(state['frontier'])
                else:
                    if not state['bfs_queue_deque']: state['mode'] = "DONE"; break
                    current = state['bfs_queue_deque'].popleft()
                    # BFS: Pop the next item from the queue
                    current = state['frontier'].popleft()

                state['cells_checked'] = len(state['bfs_visited'])
                
                if current == state['end_node']: 
                    state['mode'] = "DONE"
                    break
                
                for neighbor in current.get_valid_moves(state['grid']):
                    
                    cost = neighbor.weight if state['costs_enabled'] else 1
                    new_g = current.g + cost

                    if neighbor.g > new_g:
                        neighbor.g = new_g
                        neighbor.parent = current
                        
                        if neighbor not in state['bfs_visited']:
                            state['bfs_visited'].add(neighbor)
                            
                            # Unified push operation
                            if state['costs_enabled']:
                                heapq.heappush(state['bfs_queue_list'], (neighbor.g, neighbor))
                                heapq.heappush(state['frontier'], (neighbor.g, neighbor))
                            else:
                                state['bfs_queue_deque'].append(neighbor)
                                state['frontier'].append(neighbor)

        elif state['mode'] == "SOLVING_ASTAR":
            for _ in range(15):
                if state['open_set']:
                    state['cells_checked'] = len(state['closed_set'])
                    current_f, current = heapq.heappop(state['open_set'])
                    
                    if current == state['end_node']: 
                        state['mode'] = "DONE"
                        break
                    
                    state['closed_set'].add(current)
                    
                    for neighbor in current.get_valid_moves(state['grid']):
                        if neighbor in state['closed_set']: continue
                        
                        cost = neighbor.weight if state['costs_enabled'] else 1
                        temp_g = current.g + cost
                        
                        if temp_g < neighbor.g:
                            neighbor.parent = current
                            neighbor.g = temp_g
                            neighbor.h = heuristic(neighbor, state['end_node'])
                            neighbor.f = neighbor.g + neighbor.h
                            
                            # Add the neighbor to the open set. If a better path is found,
                            # this new entry will be processed first by the heap.
                            heapq.heappush(state['open_set'], (neighbor.f, neighbor))
                else: # No more cells in the open set, path not found
                    state['mode'] = "DONE"
                    break

        if state['mode'] == "DONE":
            if state['end_node'] and state['end_node'].parent: draw_path(state['end_node'], screen)

        # --- UI Drawing ---
        # This section is moved down to ensure it's the last thing drawn (top layer)

        # Top line of UI
        gen_algo_text = state['generation_algorithm'].replace("_", " ")
        
        # UI LOGIC
        display_algo = "Ready"
        if state['mode'] == "SOLVING_BFS":
             display_algo = "Dijkstra" if state['costs_enabled'] else "BFS"
        elif state['mode'] == "SOLVING_ASTAR":
             display_algo = "A* (Greedy)"
        elif state['mode'] == "DONE":
             # Fallback logic for when it's done to keep showing the last used
             if state['closed_set']: display_algo = "A* (Greedy)"
             elif state['bfs_visited']: display_algo = "Dijkstra" if state['costs_enabled'] else "BFS"

        status_text = f"GEN: {gen_algo_text} | MODE: {state['mode']} | ALG: {display_algo} | CELLS: {state['cells_checked']}"
        
        # Display path cost when done
        if state['mode'] == "DONE" and state['end_node'] and state['end_node'].parent:
            status_text += f" | PATH COST: {state['end_node'].g}"

        status_surface = font.render(status_text, True, BLACK, WHITE)
        status_rect = status_surface.get_rect(center=(WIDTH // 2, UI_HEIGHT // 2 - 12))
        screen.blit(status_surface, status_rect)
        
        # Bottom line of UI
        costs_status = "ON" if state['costs_enabled'] else "OFF"
        controls_text = f"Set: L/R Click | R: Reset | Gen: 1(BT), 2(Prim) | D: Costs({costs_status}) | Solve: Space(BFS), A(A*)"
        controls_surface = font.render(controls_text, True, BLACK, WHITE)
        controls_rect = controls_surface.get_rect(center=(WIDTH // 2, UI_HEIGHT // 2 + 12))
        screen.blit(controls_surface, controls_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()