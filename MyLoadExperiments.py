import os
import logging
import warnings
import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths
from scipy.integrate import simpson as simps

from MyLoadData import LoadFiles


# %% --------------------------------------------------------------------------
# INITIAL CONFIGURATION
# -----------------------------------------------------------------------------

REQ_LEVEL = 25
logging.addLevelName(REQ_LEVEL, "REQ")

def req(self, message, *args, **kwargs):
    if self.isEnabledFor(REQ_LEVEL):
        self._log(REQ_LEVEL, message, args, **kwargs)

logging.Logger.req = req
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    
mpl.use("Qt5Agg")
plt.close("all")
plt.ion()


# %% --------------------------------------------------------------------------
# PATH SELECTION
# -----------------------------------------------------------------------------

def select_paths():
    """
    Opens file dialogs to select directories and configuration files.

    Returns
    -------
    df_exps : pd.DataFrame or None
        Experiment metadata with merged load info.
    reports_dir : str or None
        Path to the directory where PDF reports are stored.
    datasets_dir : str or None
        Path to the directory where processed datasets are stored.
    """
    
    def ask_directory(prompt):
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        directory = filedialog.askdirectory(title=prompt)
        return os.path.normpath(directory) if directory else None
    
    def ask_file(prompt):
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(title=prompt)
        return os.path.normpath(file_path) if file_path else None
    
    logger.req("Please provide the experimet folder location.")
    exps_dir = ask_directory("Select Experiments Directory")
    if not exps_dir:
        logger.info("Directory selection canceled.")
        return None, None, None
    
    rawdata_dir = os.path.join(exps_dir, "RawData")
    reports_dir = os.path.join(exps_dir, "Reports")
    datasets_dir = os.path.join(exps_dir, "DataSets")
    
    logger.req("Pleae provide the Experiments Description file location.")
    exps_file = ask_file("Select Experiments Description Excel File")
    if not exps_file:
        logger.info("Experiment file selection canceled.")
        return None, None, None
    
    logger.req("Please provide the Loads Description file location.")
    loads_file = ask_file("Select Loads Description Excel File")
    if not loads_file:
        logger.info("Loads file selection canceled.")
        
    df_exps = pd.read_excel(exps_file)
    df_loads = pd.read_excel(loads_file)
    
    # Ask for TribuId
    logger.req("Please provide a TribuId value.")
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    tribu_id = simpledialog.askstring(title="TribuId Selection",
                                      prompt="Enter a valid TribuId value:")
    
    if not tribu_id:
        logger.info("TribuId selection canceled.")
        return None, None, None
    elif tribu_id in df_exps['TribuId'].dropna().unique():
        logger.info(f"TribuId value correctly saved: {tribu_id}")
        df_exps = df_exps[df_exps['TribuId'] == tribu_id]
    else:
        logger.error(f"Invalid TribuId value: {tribu_id}")
        return None, None, None
    
    # Merge with load info
    load_fields = ("Req", "Gain", "Ceq")
    for field in load_fields:
        if field not in df_exps.columns:
            df_exps.insert(2, field, None)
    
    for idx, row in df_exps.iterrows():
        if row.RloadId in df_loads.RloadId.values:
            for field in load_fields:
                df_exps.loc[idx, field] = df_loads.loc[df_loads.RloadId == row.RloadId, field].values[0]
        elif row.RloadId == "ElectrodeImpedance":
            logger.warning(f"Load {row.RloadId} is Electrode Impedance. Assigned to 80 kOhms.")
            df_exps.loc[idx, load_fields] = [80e3, 1, None]
        else:
            logger.warning(f"Load {row.RloadId} not found. Assigned open circuit.")
            df_exps.loc[idx, load_fields] = [float("inf"), 1, None]
        
        # Check file existence
        daq_file = os.path.join(rawdata_dir, row.DaqFile)
        motor_file = os.path.join(rawdata_dir, row.MotorFile)
        
        if os.path.isfile(daq_file):
            df_exps.loc[idx, "DaqFile"] = daq_file
        else:
            logger.warning(f"File {daq_file} not found. Experiment {row.ExpId} dropped.")
            df_exps.drop(idx, inplace=True)
            continue
        
        if os.path.isfile(motor_file):
            df_exps.loc[idx, "MotorFile"] = motor_file
        else:
            logger.warning(f"File {motor_file} not found. Experiment {row.ExpId} dropped.")
            df_exps.drop(idx, inplace=True)
            continue
    
    return df_exps, reports_dir, datasets_dir


# %% --------------------------------------------------------------------------
# CYCLE ANALYSIS
# -----------------------------------------------------------------------------

def analyze_cycle(cycle, req_value):
    """Analyzes a single cycle and returns metrics as dict."""
    
    imax = cycle.Voltage.idxmax()
    imin = cycle.Voltage.idxmin()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos_width = peak_widths(cycle.Voltage.values, [imax], rel_height=0.5)[0][0] * cycle.Time.diff().mean()
        neg_width = peak_widths(cycle.Voltage.values, [imin], rel_height=0.5)[0][0] * cycle.Time.diff().mean()
    
    cycle["Power"] = cycle["Voltage"]**2 / req_value
    
    cycle_pos = cycle[cycle["Voltage"] > 0].copy()
    cycle_neg = cycle[cycle["Voltage"] < 0].copy()
    cycle_neg["Voltage"] = -cycle_neg["Voltage"]
    
    return {
        "VoltageMax": cycle.loc[imax, "Voltage"],
        "VoltageMin": cycle.loc[imin, "Voltage"],
        "PosPeakWidth": pos_width,
        "NegPeakWidth": neg_width,
        "PosEnergy": simps(cycle_pos["Power"], x=cycle_pos["Time"]),
        "NegEnergy": simps(cycle_neg["Power"], x=cycle_neg["Time"]),
        }


# %% --------------------------------------------------------------------------
# EXPERIMENT ANALYSIS
# -----------------------------------------------------------------------------

def analyze_experiment(row, df_data, cycles_list, pdf):
    """Analyzes a single experiment and append figures to PDF."""
    
    exp_df = pd.DataFrame()
    
    # Figures
    fig_sep, ax_volt_sep = plt.subplots()
    ax_pos_sep = ax_volt_sep.twinx()
    ax_volt_sep.plot(df_data["Time"], df_data["Voltage"], color="red", label="Voltage")
    ax_pos_sep.plot(df_data["Time"], df_data["Position"], color="black", linestyle="--", label="Position")
    
    fig_sup, ax_volt_sup = plt.subplots()
    ax_pos_sup = ax_volt_sup.twinx()
    first = True
    
    for cy_idx, cycle in cycles_list:
        t_rel = cycle["Time"] - cycle["Time"].iloc[0]
        ax_volt_sup.plot(t_rel, cycle["Voltage"], color="red", label="Voltage" if first else None)
        ax_pos_sup.plot(t_rel, cycle["Position"], color="black", linestyle="--", label="Position" if first else None)
        first = False
        
        ax_volt_sep.axvline(x=cycle["Time"].iloc[-1], color="yellow", linestyle=":")
        
        metrics = analyze_cycle(cycle, row.Req)
        metrics["cy_idx"] = cy_idx
        metrics["TotEnergy"] = metrics["PosEnergy"] + metrics["NegEnergy"]
        exp_df = pd.concat([exp_df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Save figures to PDF
    ax_volt_sup.set_xlabel("Time per cycle (s)")
    ax_volt_sup.set_ylabel("Voltage (V)", color="red")
    ax_pos_sup.set_ylabel("Position (mm)", color="black")
    ax_volt_sup.set_title(f"Exp: {row.ExpId}, Tribu: {row.TribuId}, Req: {row.Req/10e6:.0f}M")
    ax_volt_sup.grid(True)
    fig_sup.legend(loc="upper right")
    fig_sup.tight_layout()
    pdf.savefig(fig_sup)
    plt.close(fig_sup)
    
    ax_volt_sep.set_xlabel("Time (s)")
    ax_volt_sep.set_ylabel("Voltage (V)", color="red")
    ax_pos_sep.set_ylabel("Position (mm)", color="black")
    ax_volt_sep.set_title(f"Exp: {row.ExpId}, Tribu: {row.TribuId}, Req: {row.Req/10e6:.0f}M")
    ax_volt_sep.grid(True)
    fig_sep.legend(loc="upper right")
    fig_sep.tight_layout()
    pdf.savefig(fig_sep)
    plt.close(fig_sep)
    
    return exp_df
    

# %% --------------------------------------------------------------------------
# SUMMARY PLOTS
# -----------------------------------------------------------------------------

def generate_summary_plots(summary_df, pdf):
    """Generates summary plots across exps and saves them into PDF."""
    
    req = summary_df["Req"]
    tribu_id = summary_df["TribuId"].unique()[0]
    
    # Power vs Req
    plt.figure()
    power = summary_df["AvgTotEnergy"] / summary_df["Duration"]
    power_err = np.sqrt(summary_df["VarTotEnergy"] / summary_df["Duration"])
    plt.plot(req, power, label="Power", color="green")
    plt.fill_between(req, power - power_err, power + power_err, color="green", alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Power (W)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Energy vs Req
    plt.figure()
    for energy_type, color in [("Pos", "red"), ("Neg", "blue"), ("Tot", "green")]:
        avg = summary_df[f"Avg{energy_type}Energy"]
        err = np.sqrt(summary_df[f"Var{energy_type}Energy"])
        plt.plot(req, avg, label=f"{energy_type}Energy", color=color)
        plt.fill_between(req, avg - err, avg + err, color=color, alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Energy (J)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Voltage vs Req
    plt.figure()
    for peak_type, color in [("Max", "red"), ("Min", "blue")]:
        avg = summary_df[f"AvgVoltage{peak_type}"]
        err = np.sqrt(summary_df[f"VarVoltage{peak_type}"])
        plt.plot(req, avg, label=f"Voltage{peak_type}", color=color)
        plt.fill_between(req, avg - err, avg + err, color=color, alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Voltage (V)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Peak Widths vs Req
    plt.figure()
    for polarity, color in [("Pos", "red"), ("Neg", "blue")]:
        avg = summary_df[f"Avg{polarity}PeakWidth"]
        err = np.sqrt(summary_df[f"Var{polarity}PeakWidth"])
        plt.plot(req, avg, label=f"{polarity}PeakWidth", color=color)
        plt.fill_between(req, avg - err, avg + err, color=color, alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Peak Widths (s)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# %% --------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    df_exps, reports_dir, datasets_dir = select_paths()
    
    if df_exps is None:
        return
    
    plt.ioff()
    exps_summary = []
    tribu_id = df_exps.TribuId.unique()[0]
    
    pdf_path = os.path.join(reports_dir, f"LoadReports-{tribu_id}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for exp_idx, row in df_exps.iterrows():
            logger.info(f"\nProcessing: {row.ExpId}")
            df_data, cycles_list = LoadFiles(row.MotorFile, row.DaqFile)
            
            if df_data is None:
                logger.warning(f"Data files {row.MotorFile} and {row.DaqFile} could not been loaded. Experiment {row.ExpId} dropped.")
                continue
            
            if np.all(df_data.Current.isna()):
                logger.warning(f"Column Current not found in experiment {row.ExpId}. Cannot verify Ohm's Law.")
            else:
                i_theo = df_data["Voltage"] / row.Req
                ratio = df_data["Current"] / i_theo
                tolerance = np.abs(1 - ratio).max()
                logger.info(f"Experiment {row.ExpId}: Ohm's Law satisfied within {100*tolerance:.0f}% tolerance.")
                
            exp_df = analyze_experiment(row, df_data, cycles_list, pdf)
            
            exp_summary = {
                "ExpId": row.ExpId,
                "TribuId": row.TribuId,
                "Date": row.Date,
                "Temperature": row.Temperature,
                "Humidity": row.Humidity,
                "Req": row.Req,
                "NumCycles": len(cycles_list),
                "Duration": df_data.Time.iloc[-1],
                "AvgVoltageMax": exp_df.VoltageMax.mean(),
                "VarVoltageMax": exp_df.VoltageMax.var(),
                "AvgVoltageMin": exp_df.VoltageMin.mean(),
                "VarVoltageMin": exp_df.VoltageMin.var(),
                "AvgPosPeakWidth": exp_df.PosPeakWidth.mean(),
                "VarPosPeakWidth": exp_df.PosPeakWidth.var(),
                "AvgNegPeakWidth": exp_df.NegPeakWidth.mean(),
                "VarNegPeakWidth": exp_df.NegPeakWidth.var(),
                "AvgPosEnergy": exp_df.PosEnergy.mean(),
                "VarPosEnergy": exp_df.PosEnergy.var(),
                "AvgNegEnergy": exp_df.NegEnergy.mean(),
                "VarNegEnergy": exp_df.NegEnergy.var(),
                "AvgTotEnergy": exp_df.TotEnergy.mean(),
                "VarTotEnergy": exp_df.TotEnergy.var()
            }
            exps_summary.append([exp_summary, exp_df])
        
        # Save datasets
        summary_df = pd.DataFrame([d[0] for d in exps_summary])
        excel_path = os.path.join(datasets_dir, f"DataSets-{tribu_id}.xlsx")
        
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            for dicc, df in exps_summary:
                sheet_name = f"Exp_{dicc['ExpId']}"[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Generate summary plots
        generate_summary_plots(summary_df, pdf)
        
        pdf.close()
        logger.info(f"/nProcessing completed. PDF saved to {pdf_path} and Excel to {excel_path}.")


if __name__ == "__main__":
    main()










