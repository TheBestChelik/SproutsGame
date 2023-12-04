from tkinter import *
from tkinter import messagebox, colorchooser
from tkinter import ttk
import numpy as np
from SproutGame.Board import Board
from SproutGame.modules.geometry import generate_spot_coordinates
from SproutGame.primitives import Spot, Vector
from SproutGame.resources.constants import Color, LineStyle, VERTEX_SIZE, EDGE_WIDTH, PATH_WIDTH, CANVAS_SIZE, LOADING_ANIMATION_PATH
from SproutGame.resources.GUI_Constants import *
import threading
from PIL import Image, ImageSequence, ImageTk
import time


class Game:
    def __init__(self) -> None:
        self.time_based_game = False
        self.players_number = 0
        self.spots_number = 0
        self.players = []
        self.root = Tk()
        self.init_main_menu()
        self.root.mainloop()

    def init_main_menu(self):

        self.root.title(MAIN_MENU_TITLE)
        self.root.geometry(MAIN_MENU_GEOMETRY)

        self.root.configure(bg=BACKGROUND_COLOR)

        validate_numeric = self.root.register(self.validate_numeric_input)

        # Create and style labels
        label_players = Label(
            self.root, text=NUMBER_PLAYERS_LABEL, font=TEXT_FONT, bg=BACKGROUND_COLOR)
        label_players.pack(pady=10)

        # Style entry fields
        entry_style = {"font": TEXT_FONT, "width": 15}
        entry_players = Entry(self.root, **entry_style, validate='key',
                              validatecommand=(validate_numeric, "%P"))
        entry_players.pack(pady=10)

        label_spots = Label(
            self.root, text=NUMBER_SPOTS_LABEL, font=TEXT_FONT, bg=BACKGROUND_COLOR)
        label_spots.pack(pady=10)

        entry_spots = Entry(self.root, **entry_style, validate='key',
                            validatecommand=(validate_numeric, "%P"))
        entry_spots.pack(pady=10)
        time_based_game_var = BooleanVar()
        time_based_checkbutton = Checkbutton(
            self.root, text=TIME_BASED_GAME_TEXT, variable=time_based_game_var, font=TEXT_FONT, bg=BACKGROUND_COLOR)
        time_based_checkbutton.pack(pady=10)

        # Style submit button
        
        self.main_menu_error_label = Label(
            self.root, text="", font=TEXT_FONT, fg=ERROR_TEXT_COLOR, bg=BACKGROUND_COLOR)
        self.main_menu_error_label.pack(side='bottom', pady=10)

        self.submit_button = Button(
            self.root, text=OK_BUTTON_TEXT, command=lambda: self.submit_game_details(entry_players.get(), entry_spots.get(), time_based_game_var
                                                                               ), font=TEXT_FONT)
        self.submit_button.pack(side='bottom', pady=20)

    def init_color_name_menu(self, number):
        self.root.title(PLAYER_INFO_TITLE)

        # Create and style labels
        label_name = Label(
            self.root, text=f"Player {number + 1} Enter your name:")
        label_name.pack(pady=10)

        # Entry for player's name
        name_entry = Entry(self.root, width=30)
        name_entry.pack(pady=10)

        # Button to submit player's info
        self.submit_button.config(command=lambda: self.submit_player(
            name_entry.get()))

    def submit_player(self, name):

        if len(name) == 0:
            self.main_menu_error_label.config(text=NAME_NOT_SET_ERROR_TEXT)
            return
        color = colorchooser.askcolor()[1]
        if not color:
            self.main_menu_error_label.config(text=COLOR_NOT_SET_ERROR_TEXT)
            return
        self.main_menu_error_label.config(text="")
        self.players.append((name, color))
        if len(self.players) != self.players_number:
            self.clear_elements(
                exception=[self.submit_button, self.main_menu_error_label])
            self.init_color_name_menu(len(self.players))
            return
        self.clear_elements()
        self.init_game_stage()

        self.run()

    def submit_game_details(self, player_num_text, spot_num_text, time_based_var):

        self.time_based_game = time_based_var.get()
        if len(player_num_text) == 0:
            self.main_menu_error_label.config(text=PLAYER_NUMBER_NOT_SET_ERROR_TEXT)
            return
        if len(spot_num_text) == 0:
            self.main_menu_error_label.config(text=SPOT_NUMBER_NOT_SET_ERROR_TEXT)
            return
        self.players_number = int(player_num_text)
        self.spots_number = int(spot_num_text)
        if self.players_number < 2:
            self.main_menu_error_label.config(
                text=NOT_ENOUGH_PLAYERS_ERROR_TEXT)
            return
        if self.spots_number < 1:
            self.main_menu_error_label.config(
                text=NOT_ENOUGH_SPOTS_ERROR_TEXT)
            return
        if self.players_number > 10:
            self.main_menu_error_label.config(text=TOO_MANY_PLAYERS_ERROR_TEXT)
            return
        if self.spots_number > 8:
            self.main_menu_error_label.config(text=TOO_MANY_SPOTS_ERROR_TEXT)
            return
        self.main_menu_error_label.config(text="")
        self.clear_elements([self.main_menu_error_label, self.submit_button])
        self.init_color_name_menu(0)

    def init_game_stage(self):
        self.root.geometry(MAIN_GAME_GEOMETRY)
        frm = ttk.Frame(self.root)
        frm.grid()

        # Create elements
        self.main_menu_button = Button(
            frm, text=MAIN_MENU_BUTTON_TEXT, command=self.main_menu)
        self.main_menu_button.grid(row=0, column=0, padx=10, pady=10)

        if self.time_based_game:
            self.timer_paused = True
            self.timer_label = Label(frm)
            self.timer_label.grid(row=0, column=1, padx=10, pady=10)

        self.label = Label(frm)
        self.label.grid(row=0, column=2, padx=10, pady=10)

        self.canvas_width = CANVAS_SIZE
        self.canvas_height = CANVAS_SIZE

        self.canvas = Canvas(frm,
                             width=self.canvas_width,
                             height=self.canvas_height,
                             bg=CANVAS_BACKGROUND_COLOR)
        self.canvas.grid(row=1, column=0, columnspan=3, padx=0, pady=0)

        self.canvas.bind("<Button-1>", self.move)

        self.board = None

        self.frames = []
        self.initialize_animation()

        self.animation_thread = None
        self.stop_animation = False  # Flag to indicate whether to stop the animation thread

    def clear_elements(self, exception=[]):
        # Destroy all widgets in the window
        for widget in self.root.winfo_children():
            if widget in exception:
                continue
            widget.destroy()

    def validate_numeric_input(self, new_value):
        # Validation function to allow only numeric input
        return new_value.isdigit() or new_value == ""

    def main_menu(self):
        self.clear_elements()
        self.init_main_menu()

    def initialize_animation(self):
        animated_gif = Image.open(LOADING_ANIMATION_PATH)
        for frame in ImageSequence.Iterator(animated_gif):
            frame = frame.resize(
                (int(CANVAS_SIZE*0.8), int(CANVAS_SIZE*0.8)), Image.LANCZOS)
            frame = frame.convert('RGBA')
            self.frames.append(ImageTk.PhotoImage(frame))

        self.current_frame_index = 0

    def animate(self):
        self.canvas.delete("all")
        i = 0
        while not self.stop_animation:
            self.canvas.create_image(
                CANVAS_SIZE//2, CANVAS_SIZE//2, anchor=CENTER, image=self.frames[self.current_frame_index])
            self.current_frame_index = (
                self.current_frame_index + 1) % len(self.frames)

            time.sleep(0.025)
            i += 1
            if i == 100:
                i = 0
                self.canvas.delete("all")

    def update_timer(self):
        if not self.timer_paused:
            if self.board.current_player_timer > 0:
                self.timer_label.config(
                    text=f"Time Left: {self.board.current_player_timer:.1f} seconds")
                self.board.update_current_player_timer()
                self.root.after(100, self.update_timer)
            else:
                self.timer_label.config(
                    text=f"{self.board.current_player_name}'s time is up!")
                self.board.end_of_time_based_game()

    def __determine_closest_vertex(self, x: int, y: int):
        rel_x = x/self.canvas_width
        rel_y = y/self.canvas_height
        closest_vertex = self.board.find_closest_vertex(Vector(rel_x, rel_y))
        return closest_vertex

    def draw_vertex(self, v, color=None):
        x = self.canvas_width * v.x
        y = self.canvas_height * v.y
        size = VERTEX_SIZE
        x0 = x - size // 2
        x1 = x + size / 2
        y0 = y - size / 2
        y1 = y + size / 2
        vertex_color = v.color
        if color:
            vertex_color = color
        return self.canvas.create_oval(x0, y0, x1, y1, fill=vertex_color)

    def draw_edge(self, A, B, color, linestyle=None):
        x1, y1 = self.canvas_width * A.x, self.canvas_height * A.y
        x2, y2 = self.canvas_width * B.x, self.canvas_height * B.y
        if not linestyle:
            return self.canvas.create_line(
                x1, y1, x2, y2, fill=color, width=PATH_WIDTH)
        return self.canvas.create_line(
            x1, y1, x2, y2, fill=color, dash=linestyle.value, width=EDGE_WIDTH)

    def move(self, event):

        if self.board.game_overed:
            messagebox.showinfo(GAME_OVER_TITLE, MOVE_NOT_POSSIBLE_GAME_OVER)
            return
        vertex = self.__determine_closest_vertex(event.x, event.y)
        redraw_frame, end_of_path = False, False
        if type(vertex) is Spot:
            redraw_frame, end_of_path = self.board.spot_in_path(vertex)
        else:
            redraw_frame = self.board.update_current_path(vertex)

        if redraw_frame:
            self.Display()

        if end_of_path:
            self.stop_animation = False 
            self.animation_thread = threading.Thread(target=self.animate)
            self.animation_thread.start()
            threading.Thread(target=self.perform_step).start()

    def perform_step(self):
        if self.time_based_game:
            self.timer_paused = True

        self.label.config(text=STEP_LOADING_LABEL)


        self.board.step(self.label)

        self.stop_animation = True

        self.Display()
        if self.time_based_game:
            self.timer_paused = False
            self.update_timer()

    def Display(self):
        self.canvas.delete('all')
        for path in self.board.pathes.values():
            for (A, B) in path.edges:
                self.draw_edge(A, B, path.color)

        for (A, B) in self.board.temporary_edges:
            self.draw_edge(A, B, Color.GREY.value, LineStyle.DASHED)

        visible_temporary_vetices = self.board.get_visible_verties()
        current_path = self.board.current_path
        for v in self.board.temporary_verties:
            if v in current_path:
                self.draw_vertex(v, self.board.current_player_color)
            elif v in visible_temporary_vetices:
                self.draw_vertex(v)
            else:
                self.draw_vertex(v, Color.GREY.value)

        for i in range(len(current_path)):
            A = current_path[i]
            if i < len(current_path) - 1:
                B = current_path[i+1]
                self.draw_edge(A, B, self.board.current_player_color)
            self.draw_vertex(A, self.board.current_player_color)

        for p in self.board.vertices:
            self.draw_vertex(p)

    def run(self):
        spots = set()
        field_size = 0.8
        spots_coordinates = generate_spot_coordinates(self.spots_number, field_size)
        for coordinate in spots_coordinates:
            S = Spot(coordinate[0], coordinate[1], Color.RED.value, 3)
            spots.add(S)
        
        self.board = Board(
            time_based_game=self.time_based_game,
            players=self.players,
            vertices=spots,
            optimum_length=0.1)

        self.board.step()

        self.Display()
        if self.time_based_game:
            self.timer_paused = False
            self.update_timer()


if __name__ == "__main__":
    game = Game()