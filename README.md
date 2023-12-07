# Table of Contents
- [Table of Contents](#table-of-contents)
- [Sprouts Game](#sprouts-game)
  - [Gamme rules](#gamme-rules)
    - [Standard game](#standard-game)
    - [Time-based game](#time-based-game)
  - [Gameplay](#gameplay)
  - [Implementation](#implementation)
    - [Classes and primitives](#classes-and-primitives)
      - [Vector](#vector)
      - [Vertex](#vertex)
      - [Spot](#spot)
      - [Path](#path)
      - [Board](#board)
    - [Modules](#modules)
      - [Delaunay](#delaunay)
      - [Voronoi](#voronoi)
      - [Forses](#forses)
    - [Additional recourses](#additional-recourses)
  - [Requirements](#requirements)
  - [Run the game](#run-the-game)
    - [Installation](#installation)
    - [Run](#run)

# Sprouts Game
Sprouts Game offers a deceptively simple experience for the average user, masking a complex and intricate set of algorithms beneath its surface. Developed using Python with tkinter as the user interface, the game's true depth lies in the sophistication of its algorithms.

## Gamme rules
This Sprouts game implementation features two distinct modes—standard for strategic play and time-based for added challenge—allowing players to tailor their gaming experience to their preferences.
### Standard game
In the multiplayer version of Sprouts accommodating up to 10 players, the game initiates with a set of spots, with participants taking turns drawing lines between them and introducing new spots along these lines. The gameplay follows rules where lines must not cross, new spots appear at the midpoint of lines, and no spot is allowed to exceed three connections. The victor is determined by the player who strategically executes the last move in this expanded and dynamic multiplayer setting.
### Time-based game
In the time-based mode, the Sprouts game follows the same rules as the standard version, with the key distinction being the inclusion of a limited time for each move; if a player exceeds their allotted time, they lose, and the ultimate winner is the last remaining player.

## Gameplay
![](https://github.com/TheBestChelik/SproutsGame/blob/main/img/Gameplay.gif?raw=true)

## Implementation

### Classes and primitives
In this implementation, the following classes and primitives were used.

#### Vector
Can be found in `SproutGame/resources/primitives.py`

The Vector class serves as a foundational component, encapsulating both x and y coordinates along with a set of defined mathematical operations.
#### Vertex
Can be found in `SproutGame/resources/primitives.py`

The Vertex class, another fundamental component, encompasses x and y coordinates along with a designated color attribute.
#### Spot
Can be found in `SproutGame/resources/primitives.py`

The Spot, an inherited class from Vertex, essentially extends the functionality of Vertex by incorporating a "liberties" attribute. Serving as the primary game unit, Spot plays a central role in the gameplay.
#### Path
Can be found in `SproutGame/resources/primitives.py`

The Path class holds important details about connections between spots, like the connection color and the linked vertices (Edges). An Edge is just a pair of connected vertices. Path can perform merging and separating these edges, which is useful for the game's force-rebalancing and redrawing.

#### Board
Can be found in `SproutGame/Board.py`

The Board class takes a central role in the project, overseeing the game by coordinating various processes and maintaining the game state, including vertices, connections, and paths.

### Modules

#### Delaunay
Can be found in `SproutGame/modules/geometry.py`

The Delaunay module is crucial for performing triangulation on the game field, allowing players the freedom to move between spots and establish connections (all dashed lines in the game result from triangulation). This module utilizes `Delaunator.py` for conducting the necessary computations.
#### Voronoi
Can be found in `SproutGame/modules/geometry.py`

The Voronoi module holds equal significance to Delaunay, aiming to strategically position the vertices so that players can freely draw any desired line on the field, enhancing the flexibility and creativity of gameplay.
#### Forses
Can be found in `SproutGame/modules/forces.py`

The Forces module plays a crucial role in achieving visual balance on the user's screen through force rebalancing. This module employs various forces to position vertices and spots at optimal distances from each other, ensuring an aesthetically pleasing and well-arranged image.
### Additional recourses

## Requirements

## Run the game

### Installation

### Run
