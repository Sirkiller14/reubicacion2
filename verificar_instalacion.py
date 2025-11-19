#!/usr/bin/env python3
"""
Script de Verificación de Instalación

Este script verifica que todas las dependencias estén correctamente instaladas
y que el sistema esté listo para ejecutar.

Uso:
    python verificar_instalacion.py
"""

import sys
import subprocess

def verificar_python():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        return False
    else:
        print("✅ Versión de Python correcta")
        return True


def verificar_modulo(nombre_modulo, nombre_display=None):
    """Verifica si un módulo está instalado"""
    if nombre_display is None:
        nombre_display = nombre_modulo
    
    try:
        __import__(nombre_modulo)
        print(f"✅ {nombre_display}")
        return True
    except ImportError:
        print(f"❌ {nombre_display} - NO INSTALADO")
        return False


def main():
    print("="*70)
    print("VERIFICACIÓN DE INSTALACIÓN")
    print("Sistema de Planificación Urbana con Reordenamiento")
    print("="*70)
    
    print("\n[1/3] Verificando Python...")
    python_ok = verificar_python()
    
    print("\n[2/3] Verificando dependencias principales...")
    dependencias = [
        ('osmnx', 'OSMnx'),
        ('networkx', 'NetworkX'),
        ('geopandas', 'GeoPandas'),
        ('shapely', 'Shapely'),
        ('rtree', 'Rtree'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('pymoo', 'pymoo'),
        ('folium', 'Folium'),
        ('tqdm', 'tqdm')
    ]
    
    resultados = []
    for modulo, display in dependencias:
        resultado = verificar_modulo(modulo, display)
        resultados.append((display, resultado))
    
    print("\n[3/3] Verificando versiones específicas...")
    
    # Verificar versión de OSMnx
    try:
        import osmnx as ox
        print(f"  OSMnx: v{ox.__version__}")
        if ox.__version__ < '1.0':
            print(f"  ⚠️  ADVERTENCIA: Se recomienda OSMnx >= 1.9.0")
    except Exception as e:
        print(f"  ❌ Error al verificar OSMnx: {e}")
    
    # Verificar versión de pymoo
    try:
        import pymoo
        print(f"  pymoo: v{pymoo.__version__}")
        if pymoo.__version__ < '0.6':
            print(f"  ⚠️  ADVERTENCIA: Se recomienda pymoo >= 0.6.0")
    except Exception as e:
        print(f"  ❌ Error al verificar pymoo: {e}")
    
    # Verificar NetworkX
    try:
        import networkx as nx
        print(f"  NetworkX: v{nx.__version__}")
    except Exception as e:
        print(f"  ❌ Error al verificar NetworkX: {e}")
    
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    total = len(resultados)
    correctos = sum(1 for _, ok in resultados if ok)
    
    print(f"\nDependencias instaladas: {correctos}/{total}")
    
    if correctos == total and python_ok:
        print("\n✅ TODAS LAS VERIFICACIONES PASARON")
        print("\nEl sistema está listo para usarse. Puedes ejecutar:")
        print("  python ejemplo_rapido.py")
        print("\nO ver la guía completa en:")
        print("  README.md")
        return 0
    else:
        print("\n❌ FALTAN ALGUNAS DEPENDENCIAS")
        print("\nPara instalar todas las dependencias, ejecuta:")
        print("  pip install -r requirements.txt")
        
        # Listar las que faltan
        faltantes = [nombre for nombre, ok in resultados if not ok]
        if faltantes:
            print(f"\nMódulos faltantes: {', '.join(faltantes)}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
