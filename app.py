import tkinter as tk
import tkinter.font as tkfont

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import false_discovery_control

from analytic_counter import AnalyticCounter
from epsi import EPsi
from model import Model
from u_strategy import count_ux
from ux import UX

levels = 4
u_value = 0
dx = 1 / 239
V0 = 25

class App(tk.Tk):
    def create_theme(self):
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=10, family="Arial")

        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=10, family="Arial")

        menu_font = tkfont.nametofont("TkMenuFont")
        menu_font.configure(size=10, family="Arial")

        self.configure(bg="#050505")

        self.option_add("*Background", "#111")
        self.option_add("*Foreground", "white")
        self.option_add("*Button.Background", "#111")
        self.option_add("*Button.Foreground", "white")
        self.option_add("*Entry.Background", "#111")
        self.option_add("*Entry.Foreground", "white")
        self.option_add("*Label.Background", "#111")
        self.option_add("*Label.Foreground", "white")

    def recolor_chart(self, fig, chart):
        fig.patch.set_facecolor("#050505")
        chart.set_facecolor("#111")
        chart.xaxis.label.set_color("white")
        chart.yaxis.label.set_color("white")
        chart.grid(color="#333333")
        chart.tick_params(axis='both', colors="white")
        for spine in chart.spines.values():
            spine.set_color("white")


    def __init__(self):
        super().__init__()
        self.create_theme()
        self.title("Моделлирование 1")
        self.geometry("1024x600")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        self.fig.subplots_adjust(top=0.8)
        self.fig.suptitle("Волновые функции и их производные", fontsize=14, color="white", y=0.93)
        self.recolor_chart(self.fig, self.ax1)
        self.recolor_chart(self.fig, self.ax2)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.run_button = tk.Button(self, text=r"Рассчитать E и ψ", command=self.calculate_psi)
        self.run_button.pack(side="bottom", pady=10)

        self.radio_frame = tk.Frame(self)
        self.radio_var = tk.StringVar(value="task1")
        self.task1_radio = tk.Radiobutton(self.radio_frame, text="Задание 1", variable=self.radio_var,
                                          value="task1", command=self.update_ux)
        self.task2_radio = tk.Radiobutton(self.radio_frame, text="Задание 2", variable=self.radio_var,
                                          value="task2", command=self.update_ux)

        self.task1_radio.pack(side="left")
        self.task2_radio.pack(side="right")
        self.radio_frame.pack(side="bottom")
        self.update_ux()

    def get_mode(self) -> str:
        return self.radio_var.get()

    def update_ux(self) -> None:
        mode = self.get_mode()
        if mode == "task1":
            print("Task 1")
            self.ux = count_ux(dx, lambda x: 0.0)
        elif mode == "task2":
            print("Task 2")
            self.ux = count_ux(dx, lambda x: V0 * (x ** 3))

    def calculate_first_chart(self, ux: UX, epsi: EPsi) -> None:
        self.ax1.clear()
        self.ax1.set_xlabel(r"$\frac{x}{a}$")
        self.ax1.set_ylabel(r"$\psi(\frac{x}{a})$")

        x = ux.x
        psi = epsi.psi
        E = epsi.E

        k = 0
        k_found = False
        k_eps = 0.1
        while not k_found:
            k_found = True
            for i in range(1, levels):
                if np.max(np.abs(psi[i - 1])) + E[i - 1] * k >= E[i] * k - np.max(np.abs(psi[i])):
                    k_found = False
                    k += k_eps
                    break

        for i in range(levels):
            self.ax1.plot(x, psi[i] + E[i] * k - E[0] * k, label=rf"$E_{i + 1} = {E[i]:.3f}$", linewidth=3)

        if self.get_mode() == "task1":
            analyticEPsi = AnalyticCounter(ux).count_epsi(levels)
            for i in range(levels):
                self.ax1.plot(
                    x,
                    analyticEPsi.psi[i] + analyticEPsi.E[i] * k - E[0] * k,
                    linestyle='--', color='white', label="Аналитическая функция" if i == 0 else None)

        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, None)
        self.ax1.grid(True)

        self.recolor_chart(self.fig, self.ax1)

    def calculate_second_chart(self, ux: UX, epsi: EPsi) -> None:
        self.ax2.clear()
        self.ax2.set_xlabel(r"$\frac{x}{a}$")
        self.ax2.set_ylabel(r"$\psi'(\frac{x}{a})$")

        dpsi = epsi.dpsi
        E = epsi.E
        x = ux.x

        for i in range(levels):
            self.ax2.plot(x, dpsi[i] + E[i], linewidth=3)

        if self.get_mode() == "task1":
            analyticEPsi = AnalyticCounter(ux).count_epsi(levels)
            for i in range(levels):
                self.ax2.plot(x, analyticEPsi.dpsi[i] + analyticEPsi.E[i], linestyle='--', color='white')

        self.ax2.grid(True)
        self.ax2.set_xlim(0, 1)
        self.recolor_chart(self.fig, self.ax2)

    def calculate_psi(self) -> None:
        if self.fig.legends:
            for lg in self.fig.legends:
                lg.remove()

        model = Model(self.ux)
        epsi = model.count_psi(levels)

        self.calculate_first_chart(self.ux, epsi)
        self.calculate_second_chart(self.ux, epsi)

        legend = self.fig.legend(loc='lower center', ncol=4)
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('#111')
        legend_frame.set_edgecolor("white")
        for text in legend.texts:
            text.set_color('white')

        plt.subplots_adjust(bottom=0.2)
        self.canvas.draw()

if __name__ == '__main__':
    app = App()
    app.mainloop()