import random
import typing
import numpy as np


class Tile:
    def __init__(self, pos_x, pos_y, value):
        self.x = pos_x
        self.y = pos_y
        self.value = value if value else 2
        self.previousPosition = None
        self.mergedFrom = None

    def save_position(self):
        self.previousPosition = [self.x, self.y]

    def update_position(self, pos_x, pos_y):
        self.x = pos_x
        self.y = pos_y

    def __repr__(self):
        return f"Tile({self.x}, {self.y}, {self.value})"


class Grid:
    def __init__(self, size):
        self.size = size
        self.cells: typing.List[typing.List[Tile | None]] = [
            [None for _ in range(size)] for _ in range(size)]

    def move_tile(self, tile: Tile, pos_x, pos_y):
        self.cells[tile.y][tile.x] = None
        self.cells[pos_y][pos_x] = tile
        tile.update_position(pos_x, pos_y)

    def get_tile(self, x, y) -> Tile:
        if self.within_bounds(x, y):
            return self.cells[y][x]
        return None

    def set_tile(self, x, y, tile):
        self.cells[y][x] = tile

    def insert_tile(self, tile):
        self.cells[tile.y][tile.x] = tile

    def remove_tile(self, tile):
        self.cells[tile.y][tile.x] = None

    def within_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size


class Game:
    def __init__(self, max_moves=1000, max_score=4000):
        self.state_size = 16
        self.action_size = 4
        self.grid_size = 4
        self.done = False
        self.score = 0
        self.moves = 0
        self.highest_tile = 2
        self.repeated_moves = 0
        self.max_moves = max_moves
        self.max_score = max_score
        self.grid: Grid = Grid(self.grid_size)
        self.reset()

    def _cells_available(self):
        return any([any([not cell for cell in row]) for row in self.grid.cells])

    def _random_available_cell(self):
        cells = []
        for x in range(4):
            for y in range(4):
                if self._cell_available(x, y):
                    cells.append([x, y])
        return random.choice(cells)

    def _add_random_tile(self):
        if not self._cells_available():
            return
        rand_x, rand_y = self._random_available_cell()
        value = 2 if random.random() < 0.9 else 4
        self.grid.insert_tile(Tile(rand_x, rand_y, value))

    def _vector(self, direction):
        match direction:
            case 0:
                return [0, -1]
            case 1:
                return [1, 0]
            case 2:
                return [0, 1]
            case 3:
                return [-1, 0]

    def _build_traversals(self, x, y):
        traversals_x = []
        traversals_y = []

        for pos in range(4):
            traversals_x.append(pos)
            traversals_y.append(pos)

        if x == 1:
            traversals_x = traversals_x[::-1]
        if y == 1:
            traversals_y = traversals_y[::-1]

        return traversals_x, traversals_y

    def _prepare_tiles(self):
        for row in self.grid.cells:
            for tile in row:
                if tile:
                    tile.mergedFrom = None
                    tile.save_position()

    def _cell_available(self, x, y):
        return not self.grid.get_tile(x, y)

    def _tile_matches_available(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                tile = self.grid.get_tile(x, y)
                if tile:
                    for direction in range(4):
                        vector = self._vector(direction)
                        cell = [x + vector[0], y + vector[1]]
                        other = self.grid.get_tile(*cell)
                        if other and other.value == tile.value:
                            return True
        return False

    def _find_farthest_position(self, cell, vector):
        previous = None
        while True:
            previous = cell
            cell = [previous[0] + vector[0], previous[1] + vector[1]]
            if not self.grid.within_bounds(*cell) or not self._cell_available(*cell):
                break

        return previous, cell

    def _positions_equal(self, first, second):
        if not first or not second:
            return False
        return first[0] == second[0] and first[1] == second[1]

    def _moves_available(self):
        return self._cells_available() or self._tile_matches_available()

    def _encode(self, x, max_exponent):
        return [1 if x == 2 **
                i else 0 for i in range(max_exponent)]

    def _conform_to_output(self):
        max_exponent = 13  # 2^13 = 8192
        grid = np.reshape(self.grid.cells, -1)
        grid = [x.value if x else 0 for x in grid]
        grid = np.array([self._encode(x, max_exponent) for x in grid])
        state = np.array(
            [np.reshape(grid, (self.grid_size, self.grid_size, -1))]
        )
        return state

    def stats(self):
        print(
            f"Score: {self.score} | Highest tile: {self.highest_tile} | Moves: {self.moves}/{self.max_moves}")

    def reset(self):
        self.grid = Grid(self.grid_size)
        self.score = 0
        self.done = False
        self.moves = 0
        self.highest_tile = 2
        self.repeated_moves = 0
        self._add_random_tile()
        self._add_random_tile()
        return self._conform_to_output()

    def step(self, action):
        # action: 0, 1, 2, 3
        # move:   up, right, down, left

        vector = self._vector(action)
        traversals_x, traversals_y = self._build_traversals(*vector)
        moved = False
        change_in_score = 0
        self.moves += 1

        self._prepare_tiles()

        for y in traversals_y:
            for x in traversals_x:
                tile = self.grid.get_tile(x, y)
                if tile:
                    if tile.value > self.highest_tile:
                        self.highest_tile = tile.value

                    farthest, next_position = self._find_farthest_position(
                        [x, y], vector)

                    next_cell = self.grid.get_tile(*next_position)

                    if next_cell and next_cell.value == tile.value and not next_cell.mergedFrom:
                        merged = Tile(*next_position, tile.value * 2)
                        merged.mergedFrom = [tile, next_cell]

                        self.grid.insert_tile(merged)
                        self.grid.remove_tile(tile)

                        tile.update_position(*next_position)

                        self.score += merged.value
                        change_in_score += merged.value

                        if self.score >= self.max_score:
                            print("You win! Max score reached!")
                            self.stats()
                            self.done = True
                    else:
                        self.grid.move_tile(tile, *farthest)
                    if not self._positions_equal([tile.x, tile.y], tile.previousPosition):
                        moved = True
        if moved:
            self._add_random_tile()
            if not self._moves_available():
                print("Game over! No more moves!")
                self.stats()
                self.done = True
        else:
            self.repeated_moves += 1

        if self.moves > self.max_moves:
            print("Game over! Max moves reached!")
            self.stats()
            self.done = True

        return self._conform_to_output(), change_in_score if moved else -8, self.done

    def render(self):
        for row in self.grid.cells:
            for tile in row:
                if not tile:
                    print(0, end=" ")
                else:
                    print(tile.value, end=" ")
            print()


if __name__ == "__main__":
    game = Game()
    game.render()

    # print("UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3")
    while True:
        action = int(input())
        game.step(action)
        game.render()
