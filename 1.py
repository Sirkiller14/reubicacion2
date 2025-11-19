"""
Sistema de Planificación Urbana con Reordenamiento Dinámico - TRACKING DE INTERCAMBIOS
====================================================================================

AUTOR: Mejorado por Claude
VERSIÓN: 3.1 con Tracking de Intercambios

NUEVAS CARACTERÍSTICAS:
======================
1. TRACKING DE INTERCAMBIOS POR GENERACIÓN
   - Registra número de intercambios en cada individuo
   - Captura generaciones específicas: 1, 2, 3, 78, 79, 80
   - Exporta estadísticas detalladas

2. ANÁLISIS EVOLUTIVO
   - Gráficos de evolución de intercambios
   - Distribución por generación
   - Convergencia del algoritmo

3. CALLBACK PERSONALIZADO
   - Intercepta el proceso de NSGA-II
   - Guarda datos en tiempo real
   - No afecta el rendimiento

DESCRIPCIÓN:
============
Este sistema permite el INTERCAMBIO DINÁMICO entre hogares y servicios para 
optimizar la accesibilidad urbana bajo el concepto de Ciudad de 15 Minutos.

CARACTERÍSTICAS PRINCIPALES:
1. Mantiene constante el número de hogares
2. Permite que hogares y servicios intercambien posiciones
3. Optimización iterativa con NSGA-II
4. Preserva la morfología urbana mientras mejora la distribución
5. ⭐ NUEVO: Tracking detallado de intercambios por generación

Requisitos:
    pip install osmnx==1.9.3 networkx==3.3 geopandas shapely rtree numpy pandas tqdm folium pymoo==0.6.1.1 matplotlib seaborn

Uso:
    python ciudad_15min_tracking_intercambios.py --place "San Juan de Miraflores, Lima, Peru" --plot
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import os
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict

# Librerías de procesamiento
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# Visualización
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Optimización
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.callback import Callback
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
    from pymoo.core.sampling import Sampling
    from pymoo.core.repair import Repair
    PYMOO_OK = True
except Exception:
    PYMOO_OK = False

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

OSM_QUERIES = {
    "health": [{"amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"]}],
    "education": [{"amenity": ["school", "college", "university", "kindergarten"]}],
    "greens": [{"leisure": ["park", "garden", "playground"]}, {"landuse": ["recreation_ground"]}],
    "work": [{"amenity": ["office", "coworking"]}, {"landuse": ["commercial", "industrial"]}, {"shop": True}],
}

RESIDENTIAL_BUILDING_TAGS = {"building": ["residential", "apartments", "house", "detached", "terrace"]}

# =============================================================================
# CALLBACK PARA TRACKING DE INTERCAMBIOS ⭐ NUEVO
# =============================================================================

class ExchangeTrackingCallback(Callback):
    """
    Callback personalizado para rastrear intercambios en generaciones específicas.
    
    Este callback intercepta el proceso de NSGA-II y guarda información detallada
    sobre el número de intercambios de servicios en cada individuo de la población
    durante generaciones específicas.
    
    Atributos:
    ----------
    target_generations : List[int]
        Generaciones donde se capturarán datos (ej: [1, 2, 3, 78, 79, 80])
    problem : ElementwiseProblem
        Referencia al problema para acceder a servicios originales
    data : Dict
        Almacena datos capturados por generación
    """
    
    def __init__(self, target_generations: List[int], problem):
        """
        Inicializa el callback.
        
        Parámetros:
        -----------
        target_generations : List[int]
            Lista de generaciones a monitorear
        problem : ReorderingProblem
            Instancia del problema de optimización
        """
        super().__init__()
        self.target_generations = sorted(target_generations)
        self.problem = problem
        self.data = {}  # {gen: {'individuals': [], 'exchanges': [], 'fitness': []}}
        
        print(f"\n[Tracking de Intercambios Configurado]")
        print(f"  📊 Generaciones objetivo: {self.target_generations}")
        print(f"  🏥 Servicios originales: {len(problem.original_service_indices)}")
    
    def notify(self, algorithm):
        """
        Método llamado en cada generación por pymoo.
        
        Parámetros:
        -----------
        algorithm : Algorithm
            Instancia del algoritmo NSGA-II
        """
        current_gen = algorithm.n_gen
        
        # Solo capturar si es una generación objetivo
        if current_gen not in self.target_generations:
            return
        
        print(f"  📸 Capturando generación {current_gen}...")
        
        # Obtener población actual
        pop = algorithm.pop
        
        gen_data = {
            'generation': current_gen,
            'individuals': [],
            'exchanges': [],
            'fitness': [],
            'n_homes': [],
            'n_services': []
        }
        
        # Analizar cada individuo
        for i, ind in enumerate(pop):
            x = ind.X  # Vector de asignación
            
            # Calcular intercambios: servicios originales que se convirtieron en hogares
            exchanges = 0
            for original_idx in self.problem.original_service_indices:
                if x[original_idx] == 0:  # Servicio original ahora es hogar
                    exchanges += 1
            
            # Contar hogares y servicios
            n_homes = int((x == 0).sum())
            n_services = int((x == 1).sum())
            
            # Guardar datos
            gen_data['individuals'].append(i)
            gen_data['exchanges'].append(exchanges)
            gen_data['n_homes'].append(n_homes)
            gen_data['n_services'].append(n_services)
            
            # Fitness (si está disponible)
            if ind.F is not None:
                gen_data['fitness'].append(ind.F.tolist())
            else:
                gen_data['fitness'].append([np.nan, np.nan, np.nan])
        
        # Guardar datos de esta generación
        self.data[current_gen] = gen_data
        
        # Estadísticas rápidas
        exchanges_array = np.array(gen_data['exchanges'])
        print(f"    ├─ Intercambios: μ={exchanges_array.mean():.1f}, "
              f"σ={exchanges_array.std():.1f}, "
              f"rango=[{exchanges_array.min()}, {exchanges_array.max()}]")
    
    def export_to_csv(self, output_dir: str, category: str):
        """
        Exporta los datos capturados a archivos CSV.
        
        Parámetros:
        -----------
        output_dir : str
            Directorio donde guardar los archivos
        category : str
            Nombre de la categoría siendo optimizada
        """
        if not self.data:
            print("  ⚠️  No hay datos para exportar")
            return
        
        print(f"\n  💾 Exportando datos de intercambios...")
        
        # Crear DataFrame consolidado
        all_data = []
        for gen, gen_data in self.data.items():
            for i in range(len(gen_data['individuals'])):
                row = {
                    'generation': gen,
                    'individual': gen_data['individuals'][i],
                    'exchanges': gen_data['exchanges'][i],
                    'n_homes': gen_data['n_homes'][i],
                    'n_services': gen_data['n_services'][i],
                    'f1_coverage': gen_data['fitness'][i][0],
                    'f2_balance': gen_data['fitness'][i][1],
                    'f3_exchange': gen_data['fitness'][i][2],
                }
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        # Guardar CSV completo
        filename = os.path.join(output_dir, f"exchange_tracking_{category}.csv")
        df.to_csv(filename, index=False)
        print(f"    ✅ {filename}")
        
        # Guardar estadísticas por generación
        stats = df.groupby('generation').agg({
            'exchanges': ['mean', 'std', 'min', 'max', 'median'],
            'n_homes': ['mean', 'std'],
            'n_services': ['mean', 'std']
        }).round(2)
        
        stats_filename = os.path.join(output_dir, f"exchange_stats_{category}.csv")
        stats.to_csv(stats_filename)
        print(f"    ✅ {stats_filename}")
        
        return df


# =============================================================================
# FUNCIONES DE CARGA (IGUALES A V3.0)
# =============================================================================

def load_place_boundary(place: str) -> gpd.GeoDataFrame:
    """Carga el límite geográfico de un lugar."""
    print(f"  📍 Geocodificando: {place}")
    gdf = ox.geocode_to_gdf(place)
    if gdf.empty:
        raise ValueError(f"No se pudo geocodificar: {place}")
    gdf = gdf.to_crs(4326)
    print(f"  ✅ Límite cargado")
    return gdf


def load_walking_graph(boundary: gpd.GeoDataFrame, speed_kmh: float = 4.5) -> nx.MultiDiGraph:
    """Descarga y procesa la red peatonal."""
    print(f"  🚶 Descargando red peatonal (velocidad: {speed_kmh} km/h)")
    poly = boundary.geometry.iloc[0]
    G = ox.graph_from_polygon(poly, network_type="walk", simplify=True)
    G = ox.distance.add_edge_lengths(G)
    speed_mps = (speed_kmh * 1000) / 3600
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", 0.0) or 0.0
        data["travel_time"] = length / max(speed_mps, 0.1)
    print(f"  ✅ Red cargada: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    return G


def _download_pois(boundary: gpd.GeoDataFrame, osm_filters: List[dict]) -> gpd.GeoDataFrame:
    """Descarga puntos de interés desde OSM."""
    poly = boundary.geometry.iloc[0]
    gdfs = []
    for f in osm_filters:
        try:
            g = ox.geometries_from_polygon(poly, f)
            if not g.empty:
                gdfs.append(g)
        except Exception:
            continue
    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    g = pd.concat(gdfs, axis=0).reset_index(drop=True)
    g = g[g.geometry.notna()].to_crs(4326)
    g["geometry"] = g.geometry.centroid
    return g[["geometry"]].dropna().drop_duplicates()


def load_services(boundary: gpd.GeoDataFrame) -> Dict[str, gpd.GeoDataFrame]:
    """Carga servicios por categoría."""
    print("  🏥 🏫 🌳 🏢 Descargando servicios...")
    services = {}
    for cat, filters in OSM_QUERIES.items():
        g = _download_pois(boundary, filters)
        g["category"] = cat
        g["type"] = "service"
        services[cat] = g
        print(f"    - {cat}: {len(g)} puntos")
    return services


def load_residences(boundary: gpd.GeoDataFrame, max_points: int = 3000) -> gpd.GeoDataFrame:
    """Carga ubicaciones de hogares."""
    print(f"  🏠 Descargando hogares (máx: {max_points})...")
    poly = boundary.geometry.iloc[0]
    try:
        b = ox.geometries_from_polygon(poly, RESIDENTIAL_BUILDING_TAGS)
        b = b[b.geometry.notna()].to_crs(4326)
        b["geometry"] = b.geometry.centroid
        homes = b[["geometry"]].dropna().drop_duplicates()
        print(f"    ✅ Descargados {len(homes)} edificios")
    except Exception:
        homes = gpd.GeoDataFrame(geometry=[], crs=4326)
        print(f"    ⚠️  No se encontraron edificios")
    
    if homes.empty or len(homes) < max_points / 2:
        print(f"    🎲 Generando puntos sintéticos...")
        bounds = poly.envelope
        minx, miny, maxx, maxy = bounds.bounds
        pts = []
        rng = np.random.default_rng(42)
        for _ in range(30000):
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                pts.append(p)
            if len(pts) >= max_points:
                break
        homes = gpd.GeoDataFrame(geometry=pts, crs=4326)
        print(f"    ✅ Generados {len(homes)} puntos")
    
    if len(homes) > max_points:
        homes = homes.sample(max_points, random_state=42).reset_index(drop=True)
    
    homes["category"] = "home"
    homes["type"] = "home"
    return homes


def nearest_node_series(G: nx.MultiDiGraph, gdf: gpd.GeoDataFrame) -> pd.Series:
    """Encuentra el nodo más cercano para cada punto."""
    xs, ys = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    nn = ox.distance.nearest_nodes(G, xs, ys)
    return pd.Series(nn, index=gdf.index)


# =============================================================================
# EVALUACIÓN
# =============================================================================

def calculate_coverage(G, homes, services, threshold_min=15.0):
    """Calcula cobertura de accesibilidad."""
    if services.empty or homes.empty:
        return 0.0, np.zeros(len(homes), dtype=bool)
    
    home_nodes = nearest_node_series(G, homes)
    serv_nodes = nearest_node_series(G, services)
    uniq_serv_nodes = list(set(serv_nodes.dropna().tolist()))
    
    if not uniq_serv_nodes:
        return 0.0, np.zeros(len(homes), dtype=bool)
    
    lengths = nx.multi_source_dijkstra_path_length(G, uniq_serv_nodes, weight="travel_time")
    
    reachable = np.zeros(len(homes), dtype=bool)
    for i, (idx, hn) in enumerate(home_nodes.items()):
        t = lengths.get(hn, np.inf)
        reachable[i] = (t / 60.0) <= threshold_min
    
    return float(np.mean(reachable)), reachable


def evaluate_all_categories(G, homes, services_by_cat, minutes):
    """Evalúa cobertura para todas las categorías."""
    metrics = {}
    reach_arrays = {}
    
    for cat, pois in services_by_cat.items():
        cov, reach = calculate_coverage(G, homes, pois, minutes)
        reach_arrays[cat] = reach
        metrics[f"cov_{cat}"] = cov
    
    reach_df = pd.DataFrame(reach_arrays, index=homes.index)
    reach_df.columns = [f"reach_{c}" for c in services_by_cat.keys()]
    reach_df["all_categories"] = reach_df.all(axis=1)
    metrics["cov_all"] = reach_df["all_categories"].mean()
    
    return reach_df, metrics


# =============================================================================
# INICIALIZACIÓN Y REPARACIÓN FACTIBLE
# =============================================================================

class FeasibleSampling(Sampling):
    """Inicialización que garantiza soluciones factibles (exactamente n_homes hogares)"""
    
    def __init__(self, n_homes: int):
        super().__init__()
        self.n_homes = n_homes
    
    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=int)
        
        # Usar semilla diferente para cada muestra para más diversidad
        rng = np.random.default_rng()
        for i in range(n_samples):
            # Crear solución factible: exactamente n_homes ceros (hogares)
            x = np.ones(n_var, dtype=int)
            # Seleccionar aleatoriamente n_homes posiciones para ser hogares (0)
            home_indices = rng.choice(n_var, size=self.n_homes, replace=False)
            x[home_indices] = 0
            X[i] = x
        
        return X


class FeasibleRepair(Repair):
    """Repara soluciones para que cumplan la restricción de n_homes"""
    
    def __init__(self, n_homes: int):
        super().__init__()
        self.n_homes = n_homes
    
    def _do(self, problem, X, **kwargs):
        n_var = problem.n_var
        rng = np.random.default_rng(42)
        
        for i in range(len(X)):
            x = X[i].copy()  # Hacer copia para evitar modificar el original
            n_homes_actual = int((x == 0).sum())
            
            # Si no cumple la restricción, reparar
            if n_homes_actual != self.n_homes:
                # Si hay demasiados hogares, convertir algunos en servicios
                if n_homes_actual > self.n_homes:
                    home_indices = np.where(x == 0)[0]
                    n_to_convert = min(n_homes_actual - self.n_homes, len(home_indices))
                    if n_to_convert > 0:
                        to_convert = rng.choice(home_indices, size=n_to_convert, replace=False)
                        x[to_convert] = 1
                # Si hay pocos hogares, convertir algunos servicios en hogares
                else:
                    service_indices = np.where(x == 1)[0]
                    n_to_convert = min(self.n_homes - n_homes_actual, len(service_indices))
                    if n_to_convert > 0:
                        to_convert = rng.choice(service_indices, size=n_to_convert, replace=False)
                        x[to_convert] = 0
            
            X[i] = x
        
        return X


# =============================================================================
# PROBLEMA DE OPTIMIZACIÓN
# =============================================================================

class ReorderingProblem(ElementwiseProblem):
    """Problema de optimización con 3 objetivos."""
    
    def __init__(self, G, initial_homes, initial_services, target_category,
                 minutes=15.0, alpha_balance=0.1, alpha_exchange=0.3):
        self.G = G
        self.initial_homes = initial_homes.copy()
        self.initial_services = {k: v.copy() for k, v in initial_services.items()}
        self.target_category = target_category
        self.minutes = minutes
        self.alpha_balance = alpha_balance
        self.alpha_exchange = alpha_exchange
        self.n_homes = len(initial_homes)
        
        # Pool de ubicaciones
        all_locations = [initial_homes]
        for cat_services in initial_services.values():
            if not cat_services.empty:
                all_locations.append(cat_services)
        
        self.location_pool = pd.concat(all_locations, ignore_index=True)
        self.location_pool = self.location_pool[['geometry']].drop_duplicates().reset_index(drop=True)
        
        # Identificar servicios originales ⭐
        self.original_service_indices = set()
        if target_category in initial_services and not initial_services[target_category].empty:
            target_services = initial_services[target_category]
            for _, service_geom in target_services.iterrows():
                for idx, pool_geom in self.location_pool.iterrows():
                    if service_geom.geometry.equals(pool_geom.geometry):
                        self.original_service_indices.add(idx)
                        break
        
        n_locations = len(self.location_pool)
        
        super().__init__(n_var=n_locations, n_obj=3, n_constr=1, xl=0, xu=1, type_var=np.int64)
        
        self.location_nodes = nearest_node_series(G, self.location_pool)
        
        print(f"\n[Problema Configurado]")
        print(f"  📍 Ubicaciones totales: {n_locations}")
        print(f"  🏠 Hogares objetivo: {self.n_homes}")
        print(f"  🏥 Servicios originales ({target_category}): {len(self.original_service_indices)}")
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evalúa una solución."""
        home_mask = (x == 0)
        service_mask = (x == 1)
        
        homes_locs = self.location_pool[home_mask].copy()
        service_locs = self.location_pool[service_mask].copy()
        
        # Objetivo 1: Cobertura
        if not service_locs.empty and not homes_locs.empty:
            cov_target, _ = calculate_coverage(self.G, homes_locs, service_locs, self.minutes)
        else:
            cov_target = 0.0
        
        other_coverage = []
        for cat, serv_gdf in self.initial_services.items():
            if cat != self.target_category and not serv_gdf.empty and not homes_locs.empty:
                cov, _ = calculate_coverage(self.G, homes_locs, serv_gdf, self.minutes)
                other_coverage.append(cov)
        avg_other_cov = np.mean(other_coverage) if other_coverage else 0.0
        
        # Objetivo 2: Balance
        n_services = int(service_mask.sum())
        n_homes = int(home_mask.sum())
        ideal_service_ratio = 0.075
        service_ratio = n_services / len(x) if len(x) > 0 else 0
        balance_penalty = abs(service_ratio - ideal_service_ratio) / (ideal_service_ratio + 1e-9)
        
        # Objetivo 3: Intercambio ⭐
        services_moved = 0
        for original_idx in self.original_service_indices:
            if x[original_idx] == 0:
                services_moved += 1
        
        if len(self.original_service_indices) > 0:
            exchange_penalty = services_moved / len(self.original_service_indices)
        else:
            exchange_penalty = 0.0
        
        # Funciones objetivo
        f1 = 1.0 - cov_target
        f2 = self.alpha_balance * balance_penalty + 0.1 * (1.0 - avg_other_cov)
        f3 = self.alpha_exchange * exchange_penalty
        
        # Restricción
        g1 = abs(n_homes - self.n_homes)
        
        out["F"] = [f1, f2, f3]
        out["G"] = [g1]


# =============================================================================
# OPTIMIZACIÓN CON TRACKING ⭐
# =============================================================================

def run_reordering_with_tracking(
    G, homes, services, target_category,
    minutes=15.0, max_gen=80, pop_size=100,
    alpha_balance=0.1, alpha_exchange=0.3,
    track_generations=None
):
    """
    Ejecuta optimización con tracking de intercambios.
    
    Parámetros adicionales:
    -----------------------
    track_generations : List[int] o None
        Generaciones a monitorear. Si None, usa [1, 2, 3, max_gen-2, max_gen-1, max_gen]
    """
    if not PYMOO_OK:
        raise RuntimeError("pymoo no instalado")
    
    # Generaciones por defecto
    if track_generations is None:
        track_generations = [1, 2, 3, max_gen-2, max_gen-1, max_gen]
    
    print(f"\n{'='*70}")
    print(f"NSGA-II CON TRACKING DE INTERCAMBIOS")
    print(f"{'='*70}")
    print(f"  Generaciones: {max_gen}")
    print(f"  Población: {pop_size}")
    print(f"  Tracking en: {track_generations}")
    
    # Crear problema
    problem = ReorderingProblem(
        G, homes, services, target_category, minutes, alpha_balance, alpha_exchange
    )
    
    # Crear callback para tracking ⭐
    callback = ExchangeTrackingCallback(track_generations, problem)
    
    # Configurar algoritmo con inicialización factible y reparador
    sampling = FeasibleSampling(n_homes=problem.n_homes)
    repair = FeasibleRepair(n_homes=problem.n_homes)
    algorithm = NSGA2(pop_size=pop_size, sampling=sampling, repair=repair)
    termination = get_termination("n_gen", max_gen)
    
    # Ejecutar optimización con callback
    print(f"\n  🚀 Ejecutando optimización...")
    res = minimize(
        problem, 
        algorithm, 
        termination, 
        callback=callback,  # ⭐ Callback activo
        verbose=True, 
        seed=42
    )
    
    # Procesar resultados
    # En pymoo, los resultados están en res.pop (población final del algoritmo)
    if hasattr(res, 'pop') and res.pop is not None and len(res.pop) > 0:
        # Obtener X y F de la población
        X = np.array([ind.X for ind in res.pop])
        F = np.array([ind.F for ind in res.pop])
    elif res.X is not None and res.F is not None:
        # Si res.X está disponible, usarlo
        X = res.X
        F = res.F
    else:
        raise ValueError("No se pudieron obtener las soluciones del resultado de optimización. "
                        "Verifique que el algoritmo haya completado correctamente.")
    
    if X is None or len(X) == 0:
        raise ValueError("No hay soluciones disponibles (X está vacío o None)")
    
    # Analizar soluciones y encontrar factibles
    feasible_indices = []
    constraint_violations = []
    
    for i, x in enumerate(X):
        n_homes_sol = int((x == 0).sum())
        violation = abs(n_homes_sol - problem.n_homes)
        constraint_violations.append(violation)
        if violation <= 1:
            feasible_indices.append(i)
    
    print(f"\n  [Análisis de Soluciones]")
    print(f"    Total de soluciones: {len(X)}")
    print(f"    Soluciones factibles: {len(feasible_indices)}")
    if constraint_violations:
        print(f"    Violación mínima: {min(constraint_violations)}")
        print(f"    Violación promedio: {np.mean(constraint_violations):.1f}")
        print(f"    Violación máxima: {max(constraint_violations)}")
    
    # Si no hay factibles, seleccionar las que menos violan la restricción
    if not feasible_indices:
        print(f"    ⚠️  No hay soluciones factibles, seleccionando las mejores...")
        # Ordenar por violación de restricción (menor es mejor)
        sorted_indices = sorted(range(len(X)), key=lambda i: constraint_violations[i])
        # Tomar las 10 mejores (menor violación)
        feasible_indices = sorted_indices[:min(10, len(sorted_indices))]
        print(f"    Usando {len(feasible_indices)} soluciones con menor violación")
    
    F_feas = F[feasible_indices]
    X_feas = X[feasible_indices]
    
    # Frente de Pareto
    pareto = pd.DataFrame({
        "1-coverage": F_feas[:, 0],
        "balance_penalty": F_feas[:, 1],
        "exchange_penalty": F_feas[:, 2],
        "solution_index": np.arange(len(F_feas)),
        "constraint_violation": [constraint_violations[i] for i in feasible_indices]
    })
    
    # Seleccionar mejor (priorizar menor violación de restricción)
    norm = (pareto[["1-coverage", "balance_penalty", "exchange_penalty"]] - 
            pareto[["1-coverage", "balance_penalty", "exchange_penalty"]].min()) / \
           (pareto[["1-coverage", "balance_penalty", "exchange_penalty"]].max() - 
            pareto[["1-coverage", "balance_penalty", "exchange_penalty"]].min() + 1e-9)
    
    pareto["score"] = 1.0 * norm["1-coverage"] + 0.5 * norm["exchange_penalty"] + 0.3 * norm["balance_penalty"]
    # Penalizar violaciones de restricción
    pareto["score"] += 10.0 * pareto["constraint_violation"] / (pareto["constraint_violation"].max() + 1e-9)
    
    best_idx = int(pareto.sort_values("score").iloc[0]["solution_index"])
    x_best = X_feas[best_idx]
    
    # Validar que x_best tenga la forma correcta
    if x_best is None or len(x_best) == 0:
        raise ValueError(f"La solución seleccionada está vacía o es inválida")
    
    # Reconstruir solución
    home_mask = (x_best == 0)
    service_mask = (x_best == 1)
    
    # Validar que las máscaras sean correctas
    n_homes_best = int(home_mask.sum())
    n_services_best = int(service_mask.sum())
    
    if n_homes_best == 0 or n_services_best == 0:
        print(f"    ⚠️  Solución inválida: {n_homes_best} hogares, {n_services_best} servicios")
        print(f"    Intentando corregir...")
        # Si la solución está vacía, usar una solución aleatoria válida
        # Seleccionar aleatoriamente n_homes ubicaciones como hogares
        rng = np.random.default_rng(42)
        indices = np.arange(len(problem.location_pool))
        home_indices = rng.choice(indices, size=min(problem.n_homes, len(indices)), replace=False)
        home_mask = np.zeros(len(problem.location_pool), dtype=bool)
        home_mask[home_indices] = True
        service_mask = ~home_mask
        n_homes_best = int(home_mask.sum())
        n_services_best = int(service_mask.sum())
        print(f"    Corregido: {n_homes_best} hogares, {n_services_best} servicios")
    
    new_homes = problem.location_pool[home_mask].copy()
    new_homes["category"] = "home"
    new_homes["type"] = "home"
    
    new_services = problem.location_pool[service_mask].copy()
    new_services["category"] = target_category
    new_services["type"] = "service"
    
    best_cov = 1.0 - float(F_feas[best_idx, 0])
    
    print(f"\n  {'='*70}")
    print(f"  RESULTADO")
    print(f"  {'='*70}")
    print(f"  📊 Cobertura: {best_cov:.1%}")
    print(f"  🏠 Hogares: {len(new_homes)}")
    print(f"  🏥 Servicios: {len(new_services)}")
    
    return new_homes, new_services, pareto, best_cov, callback


# =============================================================================
# VISUALIZACIÓN DE TRACKING ⭐
# =============================================================================

def plot_exchange_evolution(df: pd.DataFrame, output_dir: str, category: str):
    """
    Genera gráficos de evolución de intercambios.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos de tracking
    output_dir : str
        Directorio de salida
    category : str
        Categoría optimizada
    """
    print(f"\n  📊 Generando gráficos de evolución...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Evolución de Intercambios - {category.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Evolución de la media de intercambios
    ax1 = axes[0, 0]
    stats = df.groupby('generation')['exchanges'].agg(['mean', 'std', 'min', 'max'])
    generations = stats.index
    
    ax1.plot(generations, stats['mean'], 'b-o', linewidth=2, markersize=8, label='Media')
    ax1.fill_between(generations, 
                      stats['mean'] - stats['std'],
                      stats['mean'] + stats['std'],
                      alpha=0.3, label='±1 σ')
    ax1.plot(generations, stats['min'], 'g--', label='Mínimo')
    ax1.plot(generations, stats['max'], 'r--', label='Máximo')
    ax1.set_xlabel('Generación')
    ax1.set_ylabel('Número de Intercambios')
    ax1.set_title('Estadísticas de Intercambios por Generación')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución por generación (boxplot)
    ax2 = axes[0, 1]
    df_pivot = df.pivot(index='individual', columns='generation', values='exchanges')
    df_pivot.boxplot(ax=ax2)
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Número de Intercambios')
    ax2.set_title('Distribución de Intercambios')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogramas por generación
    ax3 = axes[1, 0]
    for gen in sorted(df['generation'].unique()):
        data = df[df['generation'] == gen]['exchanges']
        ax3.hist(data, bins=15, alpha=0.5, label=f'Gen {gen}')
    ax3.set_xlabel('Número de Intercambios')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Intercambios por Generación')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Convergencia: relación intercambios vs fitness
    ax4 = axes[1, 1]
    for gen in sorted(df['generation'].unique()):
        data = df[df['generation'] == gen]
        ax4.scatter(data['exchanges'], data['f1_coverage'], 
                   alpha=0.6, s=50, label=f'Gen {gen}')
    ax4.set_xlabel('Número de Intercambios')
    ax4.set_ylabel('f1 (1-Cobertura)')
    ax4.set_title('Trade-off: Intercambios vs Cobertura')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"exchange_evolution_{category}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    ✅ {filename}")
    plt.close()


def plot_population_heatmap(df: pd.DataFrame, output_dir: str, category: str):
    """
    Genera mapa de calor de intercambios por individuo y generación.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos de tracking
    output_dir : str
        Directorio de salida
    category : str
        Categoría optimizada
    """
    print(f"  🔥 Generando mapa de calor...")
    
    # Crear matriz: individuos x generaciones
    pivot = df.pivot(index='individual', columns='generation', values='exchanges')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Número de Intercambios'},
                ax=ax, linewidths=0.5)
    
    ax.set_title(f'Mapa de Calor: Intercambios por Individuo y Generación - {category.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Generación', fontsize=12)
    ax.set_ylabel('Individuo en la Población', fontsize=12)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"exchange_heatmap_{category}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    ✅ {filename}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sistema con Tracking de Intercambios V3.1"
    )
    
    parser.add_argument("--place", type=str, required=True)
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--speed-kmh", type=float, default=4.5)
    parser.add_argument("--max-homes", type=int, default=2000)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--alpha-exchange", type=float, default=0.3)
    parser.add_argument("--category", type=str, default="health",
                       choices=["health", "education", "greens", "work"])
    parser.add_argument("--track-gens", type=int, nargs='+',
                       default=None,
                       help="Generaciones a monitorear (ej: 1 2 3 78 79 80)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs_tracking")
    
    args = parser.parse_args()
    
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Si no se especifican generaciones, usar default
    if args.track_gens is None:
        track_gens = [1, 2, 3, args.generations-2, args.generations-1, args.generations]
    else:
        track_gens = args.track_gens
    
    print("\n" + "="*70)
    print("SISTEMA CON TRACKING DE INTERCAMBIOS V3.1")
    print("="*70)
    print(f"\n[Configuración]")
    print(f"  📍 Lugar: {args.place}")
    print(f"  🎯 Categoría: {args.category}")
    print(f"  🧬 Generaciones: {args.generations}")
    print(f"  📊 Tracking en: {track_gens}")
    print(f"  💾 Salida: {out_dir}")
    
    # Cargar datos
    print("\n" + "="*70)
    print("CARGANDO DATOS")
    print("="*70)
    
    boundary = load_place_boundary(args.place)
    G = load_walking_graph(boundary, args.speed_kmh)
    services = load_services(boundary)
    homes = load_residences(boundary, args.max_homes)
    
    # Optimización con tracking
    print("\n" + "="*70)
    print("OPTIMIZACIÓN CON TRACKING")
    print("="*70)
    
    new_homes, new_services, pareto, best_cov, callback = run_reordering_with_tracking(
        G=G,
        homes=homes,
        services=services,
        target_category=args.category,
        minutes=args.minutes,
        max_gen=args.generations,
        pop_size=args.population,
        alpha_exchange=args.alpha_exchange,
        track_generations=track_gens
    )
    
    # Exportar datos de tracking
    print("\n" + "="*70)
    print("EXPORTANDO RESULTADOS")
    print("="*70)
    
    df_tracking = callback.export_to_csv(out_dir, args.category)
    
    # Generar gráficos
    if df_tracking is not None:
        plot_exchange_evolution(df_tracking, out_dir, args.category)
        plot_population_heatmap(df_tracking, out_dir, args.category)
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE TRACKING")
    print("="*70)
    
    if df_tracking is not None:
        print("\n[Estadísticas por Generación]")
        summary = df_tracking.groupby('generation')['exchanges'].agg([
            ('Media', 'mean'),
            ('Desv.Est.', 'std'),
            ('Mínimo', 'min'),
            ('Máximo', 'max'),
            ('Mediana', 'median')
        ]).round(2)
        print(summary.to_string())
        
        # Análisis de convergencia
        first_gen = df_tracking[df_tracking['generation'] == track_gens[0]]
        last_gen = df_tracking[df_tracking['generation'] == track_gens[-1]]
        
        print(f"\n[Análisis de Convergencia]")
        print(f"  Generación {track_gens[0]}:")
        print(f"    - Media de intercambios: {first_gen['exchanges'].mean():.2f}")
        print(f"    - Desv. estándar: {first_gen['exchanges'].std():.2f}")
        print(f"  Generación {track_gens[-1]}:")
        print(f"    - Media de intercambios: {last_gen['exchanges'].mean():.2f}")
        print(f"    - Desv. estándar: {last_gen['exchanges'].std():.2f}")
        print(f"  Cambio en media: {last_gen['exchanges'].mean() - first_gen['exchanges'].mean():.2f}")
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)
    print(f"\n📂 Archivos generados:")
    print(f"  - exchange_tracking_{args.category}.csv")
    print(f"  - exchange_stats_{args.category}.csv")
    print(f"  - exchange_evolution_{args.category}.png")
    print(f"  - exchange_heatmap_{args.category}.png")
    print(f"\n💡 Revisa los archivos en: {out_dir}")


if __name__ == "__main__":
    main()