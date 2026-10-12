class Waypoint: 
    def __init__(self, x: int, y: int, z: int, occupied: bool):
        self.x = x
        self.y = y
        self.z = z
        self.occupied = occupied

    def position(self):
        return(self.x, self.y, self.z)
    
    def occupy(self) -> None:
        if self.occupied:
            print(f"Waypoint at ({self.x}, {self.y}, {self.z}) is already occupied.")
        else:
            self.occupied = True
            print(f"Waypoint at ({self.x}, {self.y}, {self.z}) is now occupied.")

    def unoccupy(self) -> None:
        if self.occupied:
            self.occupied = False
            print(f"Waypoint at ({self.x}, {self.y}, {self.z}) is now unoccupied.")
        else:
            print(f"Waypoint at ({self.x}, {self.y}, {self.z}) is already unoccupied.")

    def __add__(self, other):
        if isinstance(other, Waypoint):
            return Waypoint(self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented


point1 = Waypoint(1, 2, 3, occupied=True)
print(point1.position())
print(point1.x, point1.y, point1.z, point1.unoccupy())