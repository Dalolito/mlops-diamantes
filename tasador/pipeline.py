from tasador import entrenar, monitorear, simular_drift


def main() -> None:
    # 1. ENTRENAMOS LOS CANDIDATOS Y PUBLICAMOS EL MEJOR
    print("=== 1. Entrenamiento ===")
    entrenar.main()

    # 2. SIMULAMOS UN AÑO DE TASACIONES Y VENTAS
    print("\n=== 2. Mercado de 2027 ===")
    simular_drift.main()

    # 3. MONITOREAMOS; SI HAY ALERTA, EL REENTRENAMIENTO SE DISPARA SOLO
    print("\n=== 3. Monitoreo ===")
    monitorear.main()


if __name__ == "__main__":
    main()
