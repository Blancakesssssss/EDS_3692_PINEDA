import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy.stats import skew
import os
import warnings
warnings.filterwarnings('ignore')

# EDS_TUPM253692_Pineda
# Topic        : AGR-04 - Micro-Climate Volatility
# Dataset      : Greenhouse Sensor Data (10-minute interval)
# Student      : Chester Rios Pineda | BSECE 1C | TUPM-25-3692
# Description  : Automated data pipeline for greenhouse
#                micro-climate volatility detection & analysis
# ================================================================
 
 # ── Color Palette ──────────────────────────────────────────────
COLORS = {
    'temp'     : '#E74C3C',
    'humidity' : '#3498DB',
    'co2'      : '#27AE60',
    'voc'      : '#8E44AD',
    'outdoor'  : '#E67E22',
    'morning'  : '#F4D03F',
    'afternoon': '#E74C3C',
    'bg'       : '#FAFAFA',
    'grid'     : '#ECECEC',
}
 
# ================================================================
# CLASS: GreenhousePipeline
# ================================================================
class GreenhousePipeline:
 
    def __init__(self, filepath):
        self.filepath = filepath
        self.df_raw   = None
        self.df_clean = None
        self.df       = None   # filtered dataset
 
    # ─────────────────────────────────────────────────────────────
    # MODULE 1 — DATA INGESTION
    # ─────────────────────────────────────────────────────────────
    def load_data(self):
        try:
            print("=" * 65)
            print("   AGR-04 : MICRO-CLIMATE VOLATILITY ANALYSIS PIPELINE")
            print("   Pineda, Chester Rios | TUPM-25-3692 | BSECE 1C")
            print("=" * 65)
            print("\n[1] DATA INGESTION")
            self.df_raw = pd.read_csv(
                self.filepath, sep=';', decimal=',', parse_dates=['created']
            )
            print(f"    > Dataset loaded successfully")
            print(f"    > Total rows    : {len(self.df_raw):,}")
            print(f"    > Total columns : {len(self.df_raw.columns)}")
            print(f"    > Date range    : {self.df_raw['created'].min().date()} "
                  f"to {self.df_raw['created'].max().date()}")
            print(f"    > Columns       : {list(self.df_raw.columns)}")
        except FileNotFoundError:
            print(f"    [ERROR] File not found: {self.filepath}")
            raise
        except Exception as e:
            print(f"    [ERROR] Failed to load: {e}")
            raise
 
    # ─────────────────────────────────────────────────────────────
    # MODULE 2 — DATA CLEANING
    # ─────────────────────────────────────────────────────────────
    def clean_data(self):
        try:
            print("\n[2] DATA CLEANING")
            df = self.df_raw.copy()
 
            before = len(df)
            df.drop_duplicates(inplace=True)
            print(f"    > Duplicates removed  : {before - len(df)}")
 
            null_before = df.isnull().sum().sum()
            df.dropna(inplace=True)
            print(f"    > Null rows removed   : {null_before - df.isnull().sum().sum()}")
 
            numeric_cols = [
                'greenhous_temperature_celsius',
                'greenhouse_humidity_percentage',
                'greenhouse_illuminance_lux',
                'online_temperature_celsius',
                'online_humidity_percentage',
                'greenhouse_total_volatile_organic_compounds_ppb',
                'greenhouse_equivalent_co2_ppm'
            ]
            for col in numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass
            df.dropna(subset=numeric_cols, inplace=True)
 
            df['hour'] = df['created'].dt.hour
            df['date'] = df['created'].dt.date
            self.df_clean = df.copy()
            print(f"    > Rows after cleaning : {len(self.df_clean):,}")
 
            # ── UNIQUE FILTER LOGIC ──────────────────────────────
            # Student : Pineda, Chester Rios | TUPM-25-3692
            # Filter 1: Daytime hours  06:00–18:00
            # Filter 2: Indoor temperature > 28°C (high-heat events)
            # ─────────────────────────────────────────────────────
            self.df = self.df_clean[
                (self.df_clean['hour'] >= 6) &
                (self.df_clean['hour'] <= 18) &
                (self.df_clean['greenhous_temperature_celsius'] > 28)
            ].copy().reset_index(drop=True)
 
            print(f"    > After unique filter : {len(self.df):,} rows")
            print(f"      (Daytime 06:00-18:00 AND Indoor Temp > 28°C)")
 
            os.makedirs('data', exist_ok=True)
            self.df_clean.to_csv('data/dataset_cleaned.csv', index=False)
            print(f"    > Cleaned CSV saved  : data/dataset_cleaned.csv")
 
        except Exception as e:
            print(f"    [ERROR] Cleaning failed: {e}")
            raise
 
    # ─────────────────────────────────────────────────────────────
    # MODULE 3 — STATISTICAL ANALYSIS (NumPy)
    # ─────────────────────────────────────────────────────────────
    def analyze_data(self):
        try:
            print("\n[3] STATISTICAL ANALYSIS (NumPy)")
            df       = self.df
            temp     = df['greenhous_temperature_celsius'].values
            humidity = df['greenhouse_humidity_percentage'].values
            co2      = df['greenhouse_equivalent_co2_ppm'].values
            voc      = df['greenhouse_total_volatile_organic_compounds_ppb'].values
 
            # Descriptive Statistics
            print(f"\n    --- DESCRIPTIVE STATISTICS (NumPy) ---")
            print(f"    {'Metric':<12} {'Temp(C)':>10} {'Humid(%)':>10} {'CO2(ppm)':>12} {'VOC(ppb)':>12}")
            print(f"    {'-'*58}")
            for metric, fn in [('Mean', np.mean), ('Median', np.median),
                                ('Std Dev', np.std), ('Variance', np.var),
                                ('Min', np.min), ('Max', np.max)]:
                vals = [fn(temp), fn(humidity), fn(co2), fn(voc)]
                print(f"    {metric:<12} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>12.4f} {vals[3]:>12.4f}")
 
            # Distribution Analysis
            print(f"\n    --- DISTRIBUTION ANALYSIS ---")
            print(f"    Temperature Skewness : {skew(temp):.4f}")
            print(f"    Humidity Skewness    : {skew(humidity):.4f}")
            print(f"    CO2 Skewness         : {skew(co2):.4f}")
 
            # Outlier Detection
            print(f"\n    --- OUTLIER DETECTION (IQR Method) ---")
            for name, arr in [('Temperature', temp), ('Humidity', humidity), ('CO2', co2)]:
                q1  = np.percentile(arr, 25)
                q3  = np.percentile(arr, 75)
                iqr = q3 - q1
                out = np.sum((arr < q1 - 1.5*iqr) | (arr > q3 + 1.5*iqr))
                print(f"    {name:<13} Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  Outliers={out}")
 
            # Correlation Analysis
            print(f"\n    --- CORRELATION ANALYSIS (Pearson r) ---")
            for label, a, b in [
                ('Temp vs Humidity', temp, humidity),
                ('Temp vs CO2',      temp, co2),
                ('Temp vs VOC',      temp, voc),
                ('CO2  vs VOC',      co2,  voc),
            ]:
                r = np.corrcoef(a, b)[0, 1]
                print(f"    {label:<20} r = {r:+.4f}")
 
            # Comparative Analysis
            print(f"\n    --- COMPARATIVE ANALYSIS: Morning vs Afternoon ---")
            morning   = df[df['hour'] < 12]['greenhous_temperature_celsius'].values
            afternoon = df[df['hour'] >= 12]['greenhous_temperature_celsius'].values
            print(f"    Morning   Mean Temp : {np.mean(morning):.4f} C  Std: {np.std(morning):.4f}")
            print(f"    Afternoon Mean Temp : {np.mean(afternoon):.4f} C  Std: {np.std(afternoon):.4f}")
            print(f"    Temp Difference     : {abs(np.mean(afternoon)-np.mean(morning)):.4f} C")
 
        except Exception as e:
            print(f"    [ERROR] Analysis failed: {e}")
            raise
 
    # ─────────────────────────────────────────────────────────────
    # MODULE 4 — VISUALIZATION
    # ─────────────────────────────────────────────────────────────
    def visualize_data(self):
        try:
            print("\n[4] GENERATING VISUALIZATIONS")
            os.makedirs('outputs', exist_ok=True)
            df       = self.df
            df_anim  = df.sort_values('created').head(300).reset_index(drop=True)
            temp     = df['greenhous_temperature_celsius'].values
            humidity = df['greenhouse_humidity_percentage'].values
            co2      = df['greenhouse_equivalent_co2_ppm'].values
            voc      = df['greenhouse_total_volatile_organic_compounds_ppb'].values
            morning  = df[df['hour'] < 12]['greenhous_temperature_celsius'].values
            afternoon= df[df['hour'] >= 12]['greenhous_temperature_celsius'].values
 
            # ── STATIC 1: Histograms ──────────────────────────────
            print("    Generating static_1_histogram.png ...")
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.patch.set_facecolor(COLORS['bg'])
            fig.suptitle('AGR-04 | Greenhouse Micro-Climate Volatility\n'
                         'Distribution of Key Environmental Variables',
                         fontsize=13, fontweight='bold', y=1.02)
            for ax, (arr, title, color, bins) in zip(axes, [
                (temp,     'Indoor Temperature (°C)',   COLORS['temp'],     30),
                (humidity, 'Indoor Humidity (%)',        COLORS['humidity'], 30),
                (co2,      'CO2 Concentration (ppm)',   COLORS['co2'],      40),
            ]):
                ax.set_facecolor(COLORS['bg'])
                ax.hist(arr, bins=bins, color=color, edgecolor='white',
                        linewidth=0.6, alpha=0.85)
                ax.axvline(np.mean(arr), color='black', linestyle='--',
                           linewidth=1.5, label=f'Mean: {np.mean(arr):.2f}')
                ax.axvline(np.median(arr), color='gray', linestyle=':',
                           linewidth=1.5, label=f'Median: {np.median(arr):.2f}')
                ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
                ax.set_xlabel('Value', fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(axis='y', color=COLORS['grid'], linewidth=0.7)
                ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            plt.savefig('outputs/static_1_histogram.png', dpi=150,
                        bbox_inches='tight', facecolor=COLORS['bg'])
            plt.close()
            print("    > Saved: outputs/static_1_histogram.png")
 
            # ── STATIC 2: Boxplot ─────────────────────────────────
            print("    Generating static_2_boxplot.png ...")
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor(COLORS['bg'])
            ax.set_facecolor(COLORS['bg'])
            bp = ax.boxplot(
                [morning, afternoon],
                labels=['Morning\n(06:00-12:00)', 'Afternoon\n(12:00-18:00)'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=2),
                flierprops=dict(marker='o', markersize=4, alpha=0.4)
            )
            bp['boxes'][0].set_facecolor(COLORS['morning'])
            bp['boxes'][0].set_alpha(0.8)
            bp['boxes'][1].set_facecolor(COLORS['afternoon'])
            bp['boxes'][1].set_alpha(0.8)
            ax.set_title('AGR-04 | Micro-Climate Volatility\n'
                         'Morning vs Afternoon Indoor Temperature Comparison',
                         fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel('Indoor Temperature (°C)', fontsize=10)
            ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8)
            ax.spines[['top','right']].set_visible(False)
            ax.annotate(f"Mean: {np.mean(morning):.2f}C",
                        xy=(1, np.mean(morning)), xytext=(1.25, np.mean(morning)+1.5),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
            ax.annotate(f"Mean: {np.mean(afternoon):.2f}C",
                        xy=(2, np.mean(afternoon)), xytext=(2.15, np.mean(afternoon)+1.5),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
            plt.tight_layout()
            plt.savefig('outputs/static_2_boxplot.png', dpi=150,
                        bbox_inches='tight', facecolor=COLORS['bg'])
            plt.close()
            print("    > Saved: outputs/static_2_boxplot.png")
 
            # ── STATIC 3: Scatter + Heatmap ───────────────────────
            print("    Generating static_3_scatter_heatmap.png ...")
            fig = plt.figure(figsize=(15, 6))
            fig.patch.set_facecolor(COLORS['bg'])
            gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.35)
 
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor(COLORS['bg'])
            sc  = ax1.scatter(temp, humidity, c=co2, cmap='RdYlGn_r',
                              alpha=0.4, s=12, edgecolors='none')
            cbar = plt.colorbar(sc, ax=ax1, shrink=0.85)
            cbar.set_label('CO2 (ppm)', fontsize=9)
            ax1.set_xlabel('Indoor Temperature (°C)', fontsize=10)
            ax1.set_ylabel('Indoor Humidity (%)', fontsize=10)
            ax1.set_title('Temperature vs Humidity\n(colored by CO2 level)',
                          fontsize=11, fontweight='bold')
            ax1.grid(color=COLORS['grid'], linewidth=0.6)
            ax1.spines[['top','right']].set_visible(False)
 
            ax2  = fig.add_subplot(gs[1])
            cols = ['greenhous_temperature_celsius',
                    'greenhouse_humidity_percentage',
                    'greenhouse_equivalent_co2_ppm',
                    'greenhouse_total_volatile_organic_compounds_ppb',
                    'greenhouse_illuminance_lux']
            lbls = ['Temp', 'Humidity', 'CO2', 'VOC', 'Lux']
            corr = df[cols].corr().values
            im   = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax2, shrink=0.85).set_label('Pearson r', fontsize=9)
            ax2.set_xticks(range(len(lbls)))
            ax2.set_yticks(range(len(lbls)))
            ax2.set_xticklabels(lbls, fontsize=9, rotation=30)
            ax2.set_yticklabels(lbls, fontsize=9)
            for i in range(len(lbls)):
                for j in range(len(lbls)):
                    val = corr[i, j]
                    clr = 'white' if abs(val) > 0.6 else 'black'
                    ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=9, color=clr, fontweight='bold')
            ax2.set_title('Pearson Correlation Heatmap\n(Sensor Variables)',
                          fontsize=11, fontweight='bold')
            fig.suptitle('AGR-04 | Greenhouse Micro-Climate Volatility',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.savefig('outputs/static_3_scatter_heatmap.png', dpi=150,
                        bbox_inches='tight', facecolor=COLORS['bg'])
            plt.close()
            print("    > Saved: outputs/static_3_scatter_heatmap.png")
 
            # ── STATIC 4: Dual-Axis Line Chart ────────────────────
            print("    Generating static_4_line_chart.png ...")
            df_line = df.sort_values('created').head(400).reset_index(drop=True)
            fig, ax1 = plt.subplots(figsize=(15, 5))
            fig.patch.set_facecolor(COLORS['bg'])
            ax1.set_facecolor(COLORS['bg'])
            ax1.plot(df_line.index,
                     df_line['greenhous_temperature_celsius'],
                     color=COLORS['temp'], linewidth=1.4,
                     label='Indoor Temp (°C)', alpha=0.9)
            ax1.fill_between(df_line.index,
                             df_line['greenhous_temperature_celsius'],
                             alpha=0.08, color=COLORS['temp'])
            ax1.set_xlabel('Time Step (10-min intervals)', fontsize=10)
            ax1.set_ylabel('Temperature (°C)', color=COLORS['temp'], fontsize=10)
            ax1.tick_params(axis='y', labelcolor=COLORS['temp'])
            ax1.grid(axis='x', color=COLORS['grid'], linewidth=0.6)
            ax1.spines[['top','right']].set_visible(False)
 
            ax2 = ax1.twinx()
            ax2.plot(df_line.index,
                     df_line['greenhouse_humidity_percentage'],
                     color=COLORS['humidity'], linewidth=1.4,
                     linestyle='--', label='Humidity (%)', alpha=0.9)
            ax2.fill_between(df_line.index,
                             df_line['greenhouse_humidity_percentage'],
                             alpha=0.05, color=COLORS['humidity'])
            ax2.set_ylabel('Humidity (%)', color=COLORS['humidity'], fontsize=10)
            ax2.tick_params(axis='y', labelcolor=COLORS['humidity'])
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2,
                       loc='upper right', fontsize=9, framealpha=0.7)
            fig.suptitle('AGR-04 | Greenhouse Micro-Climate Volatility\n'
                         'Indoor Temperature & Humidity Trend Over Time',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig('outputs/static_4_line_chart.png', dpi=150,
                        bbox_inches='tight', facecolor=COLORS['bg'])
            plt.close()
            print("    > Saved: outputs/static_4_line_chart.png")
 
            # ── STATIC 5: Violin Plot ─────────────────────────────
            print("    Generating static_5_violin_plot.png ...")
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor(COLORS['bg'])
            ax.set_facecolor(COLORS['bg'])
            vp = ax.violinplot([morning, afternoon], positions=[1, 2],
                               showmeans=True, showmedians=True, showextrema=True)
            for body, color in zip(vp['bodies'],
                                   [COLORS['morning'], COLORS['afternoon']]):
                body.set_facecolor(color)
                body.set_alpha(0.75)
                body.set_edgecolor('black')
                body.set_linewidth(1)
            vp['cmeans'].set_color('black')
            vp['cmeans'].set_linewidth(2)
            vp['cmedians'].set_color('white')
            vp['cmedians'].set_linewidth(2)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Morning\n(06:00-12:00)', 'Afternoon\n(12:00-18:00)'],
                               fontsize=10)
            ax.set_ylabel('Indoor Temperature (°C)', fontsize=10)
            overall_mean = np.mean(np.concatenate([morning, afternoon]))
            ax.axhline(overall_mean, color='gray', linestyle=':', linewidth=1.5,
                       label=f'Overall Mean: {overall_mean:.2f} C')
            ax.legend(fontsize=9)
            ax.grid(axis='y', color=COLORS['grid'], linewidth=0.7)
            ax.spines[['top','right']].set_visible(False)
            ax.set_title('AGR-04 | Micro-Climate Volatility\n'
                         'Temperature Distribution: Morning vs Afternoon (Violin Plot)',
                         fontsize=12, fontweight='bold', pad=10)
            plt.tight_layout()
            plt.savefig('outputs/static_5_violin_plot.png', dpi=150,
                        bbox_inches='tight', facecolor=COLORS['bg'])
            plt.close()
            print("    > Saved: outputs/static_5_violin_plot.png")
 
            # ─────────────────────────────────────────────────────
            # HELPER: generic single-line GIF maker
            # ─────────────────────────────────────────────────────
            def make_gif(fname, col, title, ylabel, color, extra_fn=None):
                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor(COLORS['bg'])
                ax.set_facecolor(COLORS['bg'])
                y_arr = df_anim[col].values
                pad   = (y_arr.max() - y_arr.min()) * 0.12
                ax.set_xlim(0, len(df_anim) - 1)
                ax.set_ylim(y_arr.min() - pad, y_arr.max() + pad)
                ax.set_xlabel('Time Step (10-min intervals)', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_title(f'AGR-04 | {title}\nPineda, Chester Rios | TUPM-25-3692',
                             fontsize=11, fontweight='bold')
                ax.grid(color=COLORS['grid'], linewidth=0.7)
                ax.spines[['top','right']].set_visible(False)
                if extra_fn:
                    extra_fn(ax)
                line, = ax.plot([], [], color=color, linewidth=1.8, alpha=0.9)
                dot,  = ax.plot([], [], 'o', color=color, markersize=7, zorder=5)
                lbl   = ax.text(0.02, 0.92, '', transform=ax.transAxes,
                                fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white', alpha=0.75))
                def init():
                    line.set_data([], [])
                    dot.set_data([], [])
                    lbl.set_text('')
                    return line, dot, lbl
                def update(frame):
                    x = np.arange(frame)
                    y = y_arr[:frame]
                    line.set_data(x, y)
                    if frame > 0:
                        dot.set_data([frame-1], [y[-1]])
                        lbl.set_text(f'{ylabel.split("(")[0].strip()}: {y[-1]:.2f}')
                    return line, dot, lbl
                ani = animation.FuncAnimation(
                    fig, update, frames=len(df_anim),
                    init_func=init, blit=True, interval=40
                )
                ani.save(f'outputs/{fname}', writer='pillow', fps=25,
                         savefig_kwargs={'facecolor': COLORS['bg']})
                plt.close()
                print(f"    > Saved: outputs/{fname}")
 
            # ── ANIMATED 1: Indoor Temperature ────────────────────
            print("    Generating animation_1_temperature.gif ...")
            make_gif('animation_1_temperature.gif',
                     'greenhous_temperature_celsius',
                     'Live Indoor Temperature Trend',
                     'Temperature (°C)', COLORS['temp'])
 
            # ── ANIMATED 2: Indoor Humidity ───────────────────────
            print("    Generating animation_2_humidity.gif ...")
            make_gif('animation_2_humidity.gif',
                     'greenhouse_humidity_percentage',
                     'Live Indoor Humidity Trend',
                     'Humidity (%)', COLORS['humidity'])
 
            # ── ANIMATED 3: CO2 Levels ────────────────────────────
            print("    Generating animation_3_co2.gif ...")
            def co2_extra(ax):
                ax.axhline(1000, color='red', linestyle='--',
                           linewidth=1.5, alpha=0.7, label='Warning: 1,000 ppm')
                ax.legend(fontsize=9)
            make_gif('animation_3_co2.gif',
                     'greenhouse_equivalent_co2_ppm',
                     'Live CO2 Levels — Greenhouse Volatility',
                     'CO2 (ppm)', COLORS['co2'], extra_fn=co2_extra)
 
            # ── ANIMATED 4: Indoor vs Outdoor Temperature ─────────
            print("    Generating animation_4_indoor_vs_outdoor.gif ...")
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor(COLORS['bg'])
            ax.set_facecolor(COLORS['bg'])
            t_in  = df_anim['greenhous_temperature_celsius'].values
            t_out = df_anim['online_temperature_celsius'].values
            all_t = np.concatenate([t_in, t_out])
            pad   = (all_t.max() - all_t.min()) * 0.12
            ax.set_xlim(0, len(df_anim) - 1)
            ax.set_ylim(all_t.min() - pad, all_t.max() + pad)
            ax.set_xlabel('Time Step (10-min intervals)', fontsize=10)
            ax.set_ylabel('Temperature (°C)', fontsize=10)
            ax.set_title('AGR-04 | Indoor vs Outdoor Temperature — Heat Amplification Effect\n'
                         'Pineda, Chester Rios | TUPM-25-3692',
                         fontsize=11, fontweight='bold')
            ax.grid(color=COLORS['grid'], linewidth=0.7)
            ax.spines[['top','right']].set_visible(False)
            ln_in,  = ax.plot([], [], color=COLORS['temp'],    linewidth=1.8,
                              label='Indoor Temp',  alpha=0.9)
            ln_out, = ax.plot([], [], color=COLORS['outdoor'], linewidth=1.8,
                              linestyle='--', label='Outdoor Temp', alpha=0.9)
            ax.legend(fontsize=9, loc='upper right')
            lbl4 = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=9,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))
            def init4():
                ln_in.set_data([], [])
                ln_out.set_data([], [])
                lbl4.set_text('')
                return ln_in, ln_out, lbl4
            def update4(frame):
                x = np.arange(frame)
                ln_in.set_data(x,  t_in[:frame])
                ln_out.set_data(x, t_out[:frame])
                if frame > 0:
                    diff = t_in[frame-1] - t_out[frame-1]
                    lbl4.set_text(
                        f'In: {t_in[frame-1]:.1f}C  |  '
                        f'Out: {t_out[frame-1]:.1f}C  |  '
                        f'Delta: {diff:+.1f}C'
                    )
                return ln_in, ln_out, lbl4
            ani4 = animation.FuncAnimation(
                fig, update4, frames=len(df_anim),
                init_func=init4, blit=True, interval=40
            )
            ani4.save('outputs/animation_4_indoor_vs_outdoor.gif',
                      writer='pillow', fps=25,
                      savefig_kwargs={'facecolor': COLORS['bg']})
            plt.close()
            print("    > Saved: outputs/animation_4_indoor_vs_outdoor.gif")
 
            # ── ANIMATED 5: VOC Levels ────────────────────────────
            print("    Generating animation_5_voc.gif ...")
            make_gif('animation_5_voc.gif',
                     'greenhouse_total_volatile_organic_compounds_ppb',
                     'Live VOC Levels — Greenhouse Volatility',
                     'VOC (ppb)', COLORS['voc'])
 
            print("\n    > ALL VISUALIZATIONS COMPLETE!")
            print("      5 Static PNGs   -> outputs/static_*.png")
            print("      5 Animated GIFs -> outputs/animation_*.gif")
 
        except Exception as e:
            print(f"    [ERROR] Visualization failed: {e}")
            raise
 
 
# ================================================================
# MAIN ENTRY POINT
# ================================================================
if __name__ == "__main__":
    pipeline = GreenhousePipeline(filepath='data/20210703_greenhouse_data.csv')
    pipeline.load_data()
    pipeline.clean_data()
    pipeline.analyze_data()
    pipeline.visualize_data()
    print("\n" + "=" * 65)
    print("   PIPELINE COMPLETE! Check your outputs/ folder.")
    print("=" * 65)