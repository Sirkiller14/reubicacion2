#!/usr/bin/env python3
"""
Script de Ejemplo Rápido - Sistema de Planificación Urbana

Este script proporciona un ejemplo simple y rápido para probar el sistema.
Usa configuración optimizada para velocidad.

Uso:
    python ejemplo_rapido.py

Nota: Cambia la variable LUGAR abajo para probar con tu distrito.
"""

import subprocess
import sys

# ============================================
# CONFIGURACIÓN - CAMBIA AQUÍ TU DISTRITO
# ============================================

LUGAR = "San Juan de Miraflores, Lima, Peru"
# Otros ejemplos:
# LUGAR = "Miraflores, Lima, Peru"
# LUGAR = "Villa El Salvador, Lima, Peru"
# LUGAR = "Lince, Lima, Peru"

# ============================================
# PARÁMETROS DE OPTIMIZACIÓN (RÁPIDO)
# ============================================

MINUTOS = 15                # Umbral de accesibilidad
MAX_HOGARES = 1000          # Reducido para velocidad
ITERACIONES = 2             # Pocas iteraciones
GENERACIONES = 40           # Reducido para velocidad
POBLACION = 60              # Reducido para velocidad
CATEGORIAS = ["health", "education"]  # Solo 2 categorías

# ============================================
# EJECUCIÓN
# ============================================

def main():
    print("="*70)
    print("EJEMPLO RÁPIDO - Sistema de Planificación Urbana")
    print("="*70)
    print(f"\nLugar: {LUGAR}")
    print(f"Configuración: RÁPIDA (para prueba)")
    print(f"  - Hogares máximos: {MAX_HOGARES}")
    print(f"  - Iteraciones: {ITERACIONES}")
    print(f"  - Categorías: {', '.join(CATEGORIAS)}")
    print("\nNota: Para análisis completo, usa más iteraciones y hogares.")
    print("="*70)
    
    respuesta = input("\n¿Deseas continuar? (s/n): ").strip().lower()
    
    if respuesta != 's':
        print("Ejecución cancelada.")
        return
    
    # Construir comando
    comando = [
        sys.executable,
        "ciudad_15min_reordenamiento.py",
        "--place", LUGAR,
        "--minutes", str(MINUTOS),
        "--max-homes", str(MAX_HOGARES),
        "--iterations", str(ITERACIONES),
        "--generations", str(GENERACIONES),
        "--population", str(POBLACION),
        "--categories"] + CATEGORIAS + [
        "--plot",
        "--output-dir", "outputs_ejemplo_rapido"
    ]
    
    print("\n" + "="*70)
    print("INICIANDO OPTIMIZACIÓN...")
    print("="*70)
    print(f"\nComando: {' '.join(comando)}\n")
    
    try:
        # Ejecutar el script principal
        resultado = subprocess.run(comando, check=True)
        
        print("\n" + "="*70)
        print("✅ EJECUCIÓN COMPLETADA EXITOSAMENTE")
        print("="*70)
        print("\nRevisa los resultados en: outputs_ejemplo_rapido/")
        print("Abre el archivo 'comparison_map.html' en tu navegador.")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print("❌ ERROR EN LA EJECUCIÓN")
        print("="*70)
        print(f"\nCódigo de error: {e.returncode}")
        print("\nPosibles soluciones:")
        print("1. Verifica que todas las dependencias estén instaladas")
        print("2. Revisa que el nombre del lugar sea correcto")
        print("3. Asegúrate de tener conexión a Internet")
        print("4. Consulta el archivo README.md para más información")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️  EJECUCIÓN INTERRUMPIDA POR EL USUARIO")
        print("="*70)
        print("\nPuedes ejecutar el script nuevamente cuando lo desees.")
    
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR INESPERADO")
        print("="*70)
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
