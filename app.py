import dearpygui.dearpygui as dpg

import ast

from joblib import load

import json

class App:
    def __init__(self) -> None:
        self.model = load('out/model.bin')
        self.scaler = load('out/scaler.bin')
        self.fields = {
            'epoch': "Epoch",
            'orbit_axis': "Orbit Axis",
            'orbit_eccentricity': "Orbit Eccentricity",
            'orbit_inclination': "Orbit Inclination",
            'perihelion_argument': "Perihelion Argument",
            'node_longitude': "Node Longitude",
            'mean_anomoly': "Mean Anomoly",
            'perihelion_distance': "Perihelion Distance",
            'aphelion_distance': "Aphelion Distance",
            'orbital_period': "Orbital Period",
            'min_orbit_intersection_distance': "Minimum Orbit Intersection Distance",
            'orbital_reference': "Orbital Reference",
            'asteroid_magnitude': "Asteroid Magnitude",
            'classification_apohele': "Apohele Asteroid",
            'classification_apollo': "Apollo Asteroid",
            'classification_aten': "Aten Asteroid",
        }

        self.map = {True: '', False: ' not'}

        dpg.create_context()
        dpg.create_viewport(title='Cosmic Threat Identifier', width=500, height=500, resizable=False)
        dpg.setup_dearpygui()

    def start(self) -> None:
        self.draw_menu()
        self.draw_window()

        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def draw_window(self) -> None:
        with dpg.window(no_title_bar=True, no_resize=True, no_move=True, height=481, width=500, pos=(0, 19)):
            for i in self.fields:
                dpg.add_input_text(label=self.fields[i], tag=i, width=140)

            dpg.add_button(label='Submit', callback=self._handle_submit)

    def draw_menu(self) -> None:
        with dpg.viewport_menu_bar():
            with dpg.menu(label='File'):
                dpg.add_menu_item(label='Load Values', callback=self._handle_load_values)
                dpg.add_menu_item(label='Quit', callback=self._exit)

    def _handle_load_values(self) -> None:
        with dpg.window(tag='path_window'):
            dpg.add_input_text(label='Path to data', width=140, tag='path')
            dpg.add_button(callback=self._handle_path_callback, label='OK')

    def _handle_path_callback(self) -> None:
        with open(dpg.get_value('path'), 'r') as h:
            data = json.loads(h.read())
        for i in data:
            dpg.set_value(i, data[i])

        dpg.delete_item('path_window')

    def _handle_submit(self) -> None:
        X = []
        for i in self.fields:
            X.append(ast.literal_eval(dpg.get_value(i)))

        X = self.scaler.transform([X])
        prediction = self.model.predict(X)[0]

        with dpg.window():
            dpg.add_text(f"The meteor does{self.map[prediction]} pose a threat.")

    def _exit(self) -> None:
        dpg.stop_dearpygui()

if __name__ == '__main__':
    app = App()
    app.start()
