#!/usr/bin/python3.13
# coding=utf-8
# %%%
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np

def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Create a GeoDataFrame from location and accident data.
    """
    
    #Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani Pozor na mozne prohozeni d a e!

    # Merge accidents and locations
    gdf = df_accidents.merge(df_locations, on='p1', how='inner')
    
    # Remove rows with unknown coodinates
    gdf = gdf[(gdf['d'] != 0) & (gdf['e'] != 0)].copy()
    
    # If d < e swap them
    swap_mask = gdf['d'] < gdf['e']
    gdf.loc[swap_mask, ['d', 'e']] = gdf.loc[swap_mask, ['e', 'd']].values
    
    # Create geometry from the coordinates
    geometry = geopandas.points_from_xy(gdf['d'], gdf['e'])
    
    # Create geodataframe with S-JTSK coordinate system
    gdf = geopandas.GeoDataFrame(gdf, geometry=geometry, crs='EPSG:5514')
    
    return gdf

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Plots accidents caused by wildlife in JHM region for 2023 and 2024
    """
    # Vykresleni grafu s nehodami se zvěří pro roky 2023-2024 

    region = "JHM"
    
    # Filter data for region and accidents caused by wildlife
    gdf_filtered = gdf[(gdf['region'] == region) & (gdf['p10'] == 4)].copy()
    
    # Convert year to datetime format
    gdf_filtered['year'] = pd.to_datetime(gdf_filtered['p2a'], dayfirst=True).dt.year
    
    # Filter years 2023 and 2024
    gdf_2023 = gdf_filtered[gdf_filtered['year'] == 2023]
    gdf_2024 = gdf_filtered[gdf_filtered['year'] == 2024]
    
    # Convert to mercator projection
    gdf_2023_mercator = gdf_2023.to_crs(epsg=3857)
    gdf_2024_mercator = gdf_2024.to_crs(epsg=3857)
    
    # create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Compute common bounds so both maps show the same area 
    all_bounds = geopandas.GeoSeries(
        pd.concat([gdf_2023_mercator.geometry, gdf_2024_mercator.geometry], ignore_index=True)
    ).total_bounds
    x_min, y_min, x_max, y_max = all_bounds

    ax1 = axes[0]
    gdf_2023_mercator.plot(ax=ax1, markersize=5, color='red', alpha=0.6, edgecolor='darkred', linewidth=0.5)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    contextily.add_basemap(ax1, source=contextily.providers.OpenStreetMap.Mapnik, zoom='auto')
    ax1.set_title('Nehody zaviněné zvěří - JHM (2023)', fontsize=14, fontweight='bold')
    
    # Remove axis labels, ticks
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = axes[1]
    gdf_2024_mercator.plot(ax=ax2, markersize=5, color='red', alpha=0.6, edgecolor='darkred', linewidth=0.5)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    contextily.add_basemap(ax2, source=contextily.providers.OpenStreetMap.Mapnik, zoom='auto')
    ax2.set_title('Nehody zaviněné zvěří - JHM (2024)', fontsize=14, fontweight='bold')

    # Remove axis labels, ticks
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')
    
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plots clustered accidents cause by alcohol in JHM region
    """
    # Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru
    
    region = "JHM"

    # Filter data for the region and alcohol
    gdf_filtered = gdf[(gdf['region'] == region) & (gdf['p11'] >= 4)].copy()

    # Convert to mercator projection for clustering
    gdf_mercator = gdf_filtered.to_crs(epsg=3857)

    # Some data was wrongly recorded as JHM and it was outside of the region, this removes them
    gdf_mercator["x"] = gdf_mercator.geometry.x
    gdf_mercator["y"] = gdf_mercator.geometry.y
    x_low, x_high = gdf_mercator["x"].quantile([0.01, 0.99])
    y_low, y_high = gdf_mercator["y"].quantile([0.01, 0.99])
    gdf_mercator = gdf_mercator[
        (gdf_mercator["x"] >= x_low) & (gdf_mercator["x"] <= x_high) &
        (gdf_mercator["y"] >= y_low) & (gdf_mercator["y"] <= y_high)
    ].copy()

    # Extract coordinates for clustering
    coords = np.column_stack([gdf_mercator["x"], gdf_mercator["y"]])

    # Number of clusters
    n_clusters = 10
    # KMeans clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    gdf_mercator['cluster'] = kmeans.fit_predict(coords)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # Save bounds before plotting
    x_min, y_min, x_max, y_max = gdf_mercator.total_bounds

    # Each cluster is represented by polygon
    cmap = plt.get_cmap('tab20')
    hulls = (
        gdf_mercator
        .dissolve(by='cluster')
        .geometry
        .apply(lambda geom: geom.convex_hull)
    )
    # Plots cluster areas
    geopandas.GeoSeries(hulls, crs=gdf_mercator.crs).plot(
        ax=ax,
        facecolor=[cmap(i % cmap.N) for i in range(len(hulls))],
        edgecolor='black',
        alpha=0.3,
        linewidth=1.0,
        label='Cluster area',
    )

    # Plot accident points
    ax.scatter(
        gdf_mercator["x"],
        gdf_mercator["y"],
        s=5,
        c='red',
        alpha=0.6,
        linewidths=0,
        zorder=4,
    )

    # Adds map
    contextily.add_basemap(ax, source=contextily.providers.OpenStreetMap.Mapnik, zoom='auto')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set title and remove axis labels/ticks
    ax.set_title('Nehody v JHM kraji s významnou mírou alkoholu', fontsize=14, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location, dpi=300, bbox_inches='tight')

    if show_figure:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")
    gdf = make_geo(df_accidents, df_locations)

    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

    # testovani splneni zadani
    import os
    assert os.path.exists("geo1.png")
    assert os.path.exists("geo2.png")