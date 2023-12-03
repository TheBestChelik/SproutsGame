from tkinter import *
from tkinter import messagebox, colorchooser
from tkinter import ttk
import numpy as np
from SproutGame.Board import Board
from SproutGame.primitives import Vertex, Spot, Vector,  Path
from SproutGame.resources.constants import Color, LineStyle, VERTEX_SIZE, EDGE_WIDTH, PATH_WIDTH, CANVAS_SIZE, LOADING_ANIMATION_PATH
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
        # self.init_game_stage()
        # self.run()
        self.root.mainloop()

    def init_main_menu(self):
        self.root.title("Main Menu")
        self.root.geometry(f"{300}x{340}")

        # Configure a custom font and background color
        custom_font = ("Helvetica", 12)
        background_color = "#f2f2f2"
        self.root.configure(bg=background_color)

        validate_numeric = self.root.register(self.validate_numeric_input)

        # Create and style labels
        label_players = Label(
            self.root, text="Enter number of players:", font=custom_font, bg=background_color)
        label_players.pack(pady=10)

        # Style entry fields
        entry_style = {"font": custom_font, "width": 15}
        entry_players = Entry(self.root, **entry_style, validate='key',
                              validatecommand=(validate_numeric, "%P"))
        entry_players.pack(pady=10)

        label_spots = Label(
            self.root, text="Enter number of spots:", font=custom_font, bg=background_color)
        label_spots.pack(pady=10)

        entry_spots = Entry(self.root, **entry_style, validate='key',
                            validatecommand=(validate_numeric, "%P"))
        entry_spots.pack(pady=10)
        time_based_game_var = BooleanVar()
        time_based_checkbutton = Checkbutton(
            self.root, text="Time-based game", variable=time_based_game_var, font=custom_font, bg=background_color)
        time_based_checkbutton.pack(pady=10)

        # Style submit button
        error_color = "#FF0000"
        self.main_menu_error_label = Label(
            self.root, text="", font=custom_font, fg=error_color, bg=background_color)
        self.main_menu_error_label.pack(side='bottom', pady=10)

        self.submit_button = Button(
            self.root, text="Submit", command=lambda: self.submit_game_details(entry_players.get(), entry_spots.get(), time_based_game_var
                                                                               ), font=custom_font)
        self.submit_button.pack(side='bottom', pady=20)

    def init_color_name_menu(self, number):
        self.root.title("Player Info")

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
            self.main_menu_error_label.config(text="enter name")
            return
        color = colorchooser.askcolor()[1]
        if not color:
            self.main_menu_error_label.config(text="Choose your color")
            return
        print(color)
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
            self.main_menu_error_label.config(text="enter number of players")
            return
        if len(spot_num_text) == 0:
            self.main_menu_error_label.config(text="enter number of spots")
            return
        self.players_number = int(player_num_text)
        self.spots_number = int(spot_num_text)
        if self.players_number < 2:
            self.main_menu_error_label.config(
                text="who are you going to play with")
            return
        if self.spots_number < 1:
            self.main_menu_error_label.config(
                text="you cant play without spots")
            return
        if self.players_number > 10:
            self.main_menu_error_label.config(text="max num players is 10")
            return
        if self.spots_number > 8:
            self.main_menu_error_label.config(text="max spot number is 8")
            return
        self.main_menu_error_label.config(text="")
        self.clear_elements([self.main_menu_error_label, self.submit_button])
        self.init_color_name_menu(0)

    def init_game_stage(self):

        self.root.geometry(f"{CANVAS_SIZE}x{CANVAS_SIZE + 40}")
        frm = ttk.Frame(self.root)
        frm.grid()

        # Create elements
        self.main_menu_button = Button(
            frm, text='Main Menu', command=self.main_menu)
        self.main_menu_button.grid(row=0, column=0, padx=10, pady=10)

        if self.time_based_game:
            self.timer_paused = True
            self.timer_label = Label(frm, text="Your middle label text")
            self.timer_label.grid(row=0, column=1, padx=10, pady=10)

        self.label = Label(frm, text="Some stupid text")
        self.label.grid(row=0, column=2, padx=10, pady=10)

        self.canvas_width = CANVAS_SIZE
        self.canvas_height = CANVAS_SIZE

        self.canvas = Canvas(frm,
                             width=self.canvas_width,
                             height=self.canvas_height,
                             bg='white')
        self.canvas.grid(row=1, column=0, columnspan=3, padx=0, pady=0)

        self.canvas.bind("<Button-1>", self.move)
        # self.canvas.bind("<Button-3>", self.cancel_move)

        self.board = None

        self.frames = []  # Store PhotoImage objects here
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
        print("Hello, I am main menu button")

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
            # self.canvas.delete("all")  # Clear the canvas
            self.canvas.create_image(
                CANVAS_SIZE//2, CANVAS_SIZE//2, anchor=CENTER, image=self.frames[self.current_frame_index])
            self.current_frame_index = (
                self.current_frame_index + 1) % len(self.frames)
            # self.root.update()  # Update the canvas
            # Change frame every 100 milliseconds
            time.sleep(0.025)
            i += 1
            if i == 100:
                i = 0
                self.canvas.delete("all")  # Clear the canvas

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
        # else:
        #     self.timer_label.config(text="Timer Paused")

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
            messagebox.showinfo("Game over", "You cant move, game overed")
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
            self.stop_animation = False  # Reset the flag before starting the animation
            self.animation_thread = threading.Thread(target=self.animate)
            self.animation_thread.start()  # Start the animation thread
            threading.Thread(target=self.perform_step).start()

    def perform_step(self):
        if self.time_based_game:
            self.timer_paused = True
        # Update the label to show that the step is in progress
        self.label.config(text="Step in progress...")

        # Perform the step operation in your separate thread
        self.board.step(self.label)
        # time.sleep(5)
        # Update the label after the step is done

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
        print("runnng")

        S1 = Spot(0.3, 0.3, Color.RED.value, 3)
        S2 = Spot(0.7, 0.7, Color.RED.value, 3)
        S3 = Spot(0.3, 0.7, Color.RED.value, 3)
        S4 = Spot(0.7, 0.3, Color.RED.value, 3)
        spots = {S1, S2, S3, S4}
        if len(self.players) == 0:
            self.time_based_game = True
            self.players = [("Bobek", Color.YELLOW.value),
                            ('Michail', Color.BLUE.value)]
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

    # S5 = Spot(0.4, 0.4, Color.RED, 3)
    # S6 = Spot(0.6, 0.6, Color.RED, 3)
    # S7 = Spot(0.4, 0.6, Color.RED, 3)
    # S8 = Spot(0.9, 0.4, Color.RED, 3)
    # spots = {S1, S2, S3, S4}
    # game.run(spots=spots)
