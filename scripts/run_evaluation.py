"""
Script de evaluación: corre el recomendador XGBoost para 20 escenarios
y exporta los top-10 resultados por escenario a un CSV.

Uso:
    PYTHONPATH=. python scripts/run_evaluation.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from app.core.database import SessionLocal
from app.services.recommendation_service import generar_recomendacion

MODE_MAP = {"driving": "driving", "walking": "walking", "cycling": "cycling"}

ESCENARIOS = [
    {
        "num": 1, "participante": "P01",
        "trabajo": "Hospital María Auxiliadora – Emergencias",
        "work_lat": -12.1610860, "work_lon": -76.9583075,
        "budget": 1400, "mode": "driving",
    },
    {
        "num": 2, "participante": "P02",
        "trabajo": "Plaza Vea Villa María del Triunfo",
        "work_lat": -12.180264316463456, "work_lon": -76.94156112554346,
        "budget": 1200, "mode": "driving",
    },
    {
        "num": 3, "participante": "P03",
        "trabajo": "Municipalidad de Chorrillos – Administración",
        "work_lat": -12.184356343645286, "work_lon": -77.00723497369239,
        "budget": 1500, "mode": "cycling",
    },
    {
        "num": 4, "participante": "P04",
        "trabajo": "SUNAT – Oficina San Isidro",
        "work_lat": -12.0962449, "work_lon": -77.0270737,
        "budget": 1800, "mode": "walking",
    },
    {
        "num": 5, "participante": "P05",
        "trabajo": "SENASA – Sede La Molina",
        "work_lat": -12.077002052767478, "work_lon": -76.94179084209684,
        "budget": 1300, "mode": "cycling",
    },
    {
        "num": 6, "participante": "P06",
        "trabajo": "Universidad Villarreal – Campus Miraflores",
        "work_lat": -12.121554558841227, "work_lon": -77.03720528071155,
        "budget": 2000, "mode": "cycling",
    },
    {
        "num": 7, "participante": "P07",
        "trabajo": "Clínica San Pablo – Surco",
        "work_lat": -12.100080490349026, "work_lon": -76.97117194531651,
        "budget": 1700, "mode": "driving",
    },
    {
        "num": 8, "participante": "P08",
        "trabajo": "Banco de la Nación – Agencia Centro Lima",
        "work_lat": -12.044659816142927,  "work_lon": -77.0308996102784,
        "budget": 1100, "mode": "cycling",
    },
    {
        "num": 9, "participante": "P09",
        "trabajo": "Colegio Fe y Alegría 17 – Docente",
        "work_lat": -12.0241727, "work_lon": -77.0766210,
        "budget": 1200, "mode": "driving",
    },
    {
        "num": 10, "participante": "P10",
        "trabajo": "Metro Chorrillos – Repositor",
        "work_lat": -12.172668338423431, "work_lon":  -77.01226238455432,
        "budget": 1000, "mode": "driving",
    },
    {
        "num": 11, "participante": "P11",
        "trabajo": "Hospital Casimiro Ulloa – Enfermería",
        "work_lat": -12.1280242, "work_lon": -77.0177447,
        "budget": 1400, "mode": "walking",
    },
    {
        "num": 12, "participante": "P12",
        "trabajo": "Ministerio de Educación – MINEDU",
        "work_lat": -12.0876787, "work_lon": -77.0059241,
        "budget": 1600, "mode": "driving",
    },
    {
        "num": 13, "participante": "P13",
        "trabajo": "UGEL 01 – Sede San Juan de Miraflores",
        "work_lat": -12.1484407, "work_lon": -76.9809023,
        "budget": 1300, "mode": "cycling",
    },
    {
        "num": 14, "participante": "P14",
        "trabajo": "BCP – Agencia Chorrillos",
        "work_lat": -12.1686135, "work_lon": -77.0263457,
        "budget": 1500, "mode": "driving",
    },
    {
        "num": 15, "participante": "P15",
        "trabajo": "Tottus Villa El Salvador – Logística",
        "work_lat": -12.19776911779604, "work_lon": -76.96391570353082,
        "budget": 1100, "mode": "cycling",
    },
    {
        "num": 16, "participante": "P16",
        "trabajo": "Essalud – Hospital Rebagliati Emergencias",
        "work_lat": -12.0782058, "work_lon": -77.0399865,
        "budget": 1800, "mode": "walking",
    },
    {
        "num": 17, "participante": "P17",
        "trabajo": "Municipalidad de Lima – Subgerencia Rentas",
        "work_lat": -12.0452582, "work_lon": -77.0312849,
        "budget": 1200, "mode": "driving",
    },
    {
        "num": 18, "participante": "P18",
        "trabajo": "Saga Falabella – Jockey Plaza",
        "work_lat": -12.0862037, "work_lon": -76.9753818,
        "budget": 1400, "mode": "cycling",
    },
    {
        "num": 19, "participante": "P19",
        "trabajo": "IE Rosa de América – Docente primaria",
        "work_lat": -12.196324027877063, "work_lon": -76.94660895226663,
        "budget": 1210, "mode": "cycling",
    },
    {
        "num": 20, "participante": "P20",
        "trabajo": "Municipalidad VES – Área Social",
        "work_lat": -12.2131656, "work_lon": -76.9371893,
        "budget": 1800, "mode": "cycling",
    },
]

OUTPUT_CSV = "data/processed/evaluacion_resultados.csv"
MAX_DISTANCE_KM = 10  # mismo default que usa la app real (onboarding y modo invitado)
MAX_DISTANCE_KM_WALKING = 2  # 10 km caminando no es un commute realista
TOP_N = 10

FIELDNAMES = [
    "Escenario", "Participante", "Trabajo simulado", "Presupuesto S/", "Modo", "Radio busqueda (km)",
    "Ranking Modelo", "Vivienda ID", "Distrito / Direccion",
    "Precio S/", "Area m2", "Habitaciones",
    "Tiempo corregido XGBoost (min)", "Tiempo OSRM crudo (min)", "Score"
]


def run():
    db = SessionLocal()
    rows = []

    for esc in ESCENARIOS:
        num = esc["num"]
        mode_osrm = MODE_MAP.get(esc["mode"], "driving")
        print(f"[{num}/20] {esc['participante']} — {esc['trabajo'][:50]}...")

        max_distance = MAX_DISTANCE_KM_WALKING if esc["mode"] == "walking" else MAX_DISTANCE_KM

        resultado = generar_recomendacion(
            work_lat=esc["work_lat"],
            work_lon=esc["work_lon"],
            budget=esc["budget"],
            mode=mode_osrm,
            db=db,
            max_distance_km=max_distance,
        )

        mensaje = resultado.get("message") or ""
        results = resultado["results"][:TOP_N]

        if not results:
            rows.append({
                "Escenario": num,
                "Participante": esc["participante"],
                "Trabajo simulado": esc["trabajo"],
                "Presupuesto S/": esc["budget"],
                "Modo": esc["mode"],
                "Radio busqueda (km)": max_distance,
                "Ranking Modelo": "",
                "Vivienda ID": "",
                "Distrito / Direccion": "",
                "Precio S/": "",
                "Area m2": "",
                "Habitaciones": "",
                "Tiempo corregido XGBoost (min)": "",
                "Tiempo OSRM crudo (min)": "",
                "Score": "",
            })
            print(f"  Sin resultados: {mensaje}")
            continue

        for rank, r in enumerate(results, 1):
            p = r["property"]
            rows.append({
                "Escenario": num,
                "Participante": esc["participante"],
                "Trabajo simulado": esc["trabajo"],
                "Presupuesto S/": esc["budget"],
                "Modo": esc["mode"],
                "Radio busqueda (km)": max_distance,
                "Ranking Modelo": rank,
                "Vivienda ID": f"VIV-{str(p.id).zfill(3)}",
                "Distrito / Direccion": f"{p.district} — {p.address}",
                "Precio S/": p.price,
                "Area m2": p.total_area_sqm,
                "Habitaciones": p.bedrooms,
                "Tiempo corregido XGBoost (min)": r["predicted_time_min"],
                "Tiempo OSRM crudo (min)": r["osrm_time_min"],
                "Score": r["match_score"],
            })

        print(f"  {len(results)} resultados. Top score: {results[0]['match_score']}")

    db.close()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV exportado en: {OUTPUT_CSV} ({len(rows)} filas)")


if __name__ == "__main__":
    run()
