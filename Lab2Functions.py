import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps
from scipy.signal import butter

"""Schritt 1: Diese Funktion findet die Peaks der Ableitung des ECG-Signals
Eingabe: ECG-Signal, Zeit
Ausgabe: Ableitung des ECG, Position der Peaks der d_ecg
"""


def decg_peaks(ecg, time):
    """Schritt 1: Finde die Peaks der Ableitung des ECG-Signals"""
    d_ecg = np.diff(ecg)  # Finde die Ableitung des ECG-Signals
    peaks_d_ecg, _ = sps.find_peaks(d_ecg)  # Peaks der d_ecg

    # Schritt 1 plotten
    plt.figure()
    plt.plot(time.iloc[0:len(time) - 1], d_ecg, color='red')
    plt.plot(time.iloc[peaks_d_ecg], d_ecg[peaks_d_ecg], "x", color='g')
    plt.xlabel('Zeit [s]')
    plt.ylabel('Ableitung der Aktivierung []')
    plt.title('R-Wellen Peaks Schritt 1: Peaks der Ableitung des EKG')
    plt.show()
    #save img
    plt.savefig('R-Wellen Peaks Schritt 1: Peaks der Ableitung des EKG.png')
    return d_ecg, peaks_d_ecg


"""Schritt 2: Diese Funktion filtert alle Peaks, die unter der Höhen-Schwelle liegen
    und nicht mindestens einen Abstand zueinander haben. 
    Eingabe: d_ecg-Signal, Position der Peaks aus decg_peaks(), Zeit,
         Höhen-Schwellen-Prozentsatz als Dezimalzahl, Abstand-Schwellen-Prozentsatz als Dezimalzahl
    Ausgabe: R-Wellen Peaks des d_ecg"""


def d_ecg_peaks(d_ecg, peaks_d_ecg, time, heightper, distanceper):
    meanpeaks_d_ecg = np.mean(d_ecg[peaks_d_ecg])  # Finde den Mittelwert der Peaks
    max_d_ecg = np.max(d_ecg)  # Finde den Maximalwert des ECG-Signals
    threshold = np.mean([meanpeaks_d_ecg,
                         max_d_ecg]) * heightper  # Finde den Mittelwert von meanpeaks_d_ecg und max_d_ecg - dies wird eine gute Schwelle für das Finden der Peaks sein. Filtert alle Peaks unten aus
    newpeaks_d_ecg, _ = sps.find_peaks(d_ecg, height=threshold)  # Finde die neuen Peaks
    newpeaks_d_ecg_t = time[newpeaks_d_ecg]
    newpeaks_d_ecg_t = newpeaks_d_ecg_t.reset_index(drop=True)
    meandistance = np.mean(np.diff(newpeaks_d_ecg))
    Rwave_peaks_d_ecg, _ = sps.find_peaks(d_ecg, height=threshold, distance=meandistance * distanceper)  #

    # Schritt 2 plotten
    plt.figure()
    plt.plot(time[0:len(time) - 1], d_ecg, color='red')
    plt.plot(time[Rwave_peaks_d_ecg], d_ecg[Rwave_peaks_d_ecg], "x", color='g')
    # plt.axhline(meanpeaks_d_ecg, color = 'b')
    # plt.axhline(max_d_ecg, color = 'b')
    thres = plt.axhline(threshold, color='black', label='Schwelle')
    plt.title('R-Wellen Peaks Schritt 2: d_EKG Peaks')
    plt.ylabel('Ableitung der Aktivierung []')
    plt.xlabel('Zeit [s]')
    plt.legend()
    plt.savefig('R-Wellen Peaks Schritt 2: d_EKG Peaks.png')
    plt.show()

    return Rwave_peaks_d_ecg


"""Schritt 3: Diese Funktion findet die R-Wellen Peaks im ursprünglichen EKG-Signal
    mit den zuvor definierten Peaks des d_ecg-Signals
    Eingabe: ECG-Signal, Ableitung des EKG-Signals,
        R-Wellen Peaks des d_ecg aus height_distance_threshold_peaks
    Ausgabe: R-Wellen Peaks"""


def Rwave_peaks(ecg, d_ecg, Rwave_peaks_d_ecg, time):
    Rwave = np.empty([len(Rwave_peaks_d_ecg) - 1])
    for i in range(0, len(Rwave)):  # für alle Peaks
        ecgrange = ecg[Rwave_peaks_d_ecg[i]:Rwave_peaks_d_ecg[
            i + 1]]  # Erstelle ein Array, das das ECG innerhalb der d_ecg_Peaks enthält
        percentage = np.round(len(ecgrange) * 0.2)
        maxvalue = np.array(
            list(np.where(ecgrange == np.max(ecgrange[0:int(percentage)]))))  # Finde den Index des Maximalwerts des ECG
        Rwave[i] = Rwave_peaks_d_ecg[i] + maxvalue[0, 0]  # Speichere diesen Index

    Rwave = Rwave.astype(np.int64)
    Rwave_t = time[Rwave]
    Rwave_t = Rwave_t.reset_index(drop=True)
    Rwave_t = Rwave_t.drop(columns=['index'])

    # Schritt 3 plotten
    fig, ax1 = plt.subplots()
    ax1.plot(time[0:len(time) - 1], d_ecg, color='r', label='Ableitung des EKG')
    ax1.set_ylabel('Ableitung der Aktivierung []')
    plt.xlabel('Zeit [s]')
    plt.title('R-Wellen Peaks Schritt 3: R-Wellen Peaks')
    ax2 = ax1.twinx()
    ax2.plot(time, ecg, color='b', label='ECG')
    ax2.plot(time[Rwave], ecg[Rwave], "x", color='g')
    ax2.set_ylabel('Aktivierung []')
    # Setze die Legende in die rechte obere Ecke des Plots
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
    plt.savefig('R-Wellen Peaks Schritt 3: R-Wellen Peaks.png')
    return Rwave_t
