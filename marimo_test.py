import marimo

__generated_with = "0.2.0"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    return mo, plt

@app.cell
def _(mo, plt):
    mo.md("Here is a plot")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    fig
    return fig,

if __name__ == "__main__":
    app.run()
