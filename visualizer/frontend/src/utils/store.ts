import { LazyStore } from "@tauri-apps/plugin-store";

// LazyStore no bloquea la ejecución: carga el archivo bajo demanda y guarda automáticamente los cambios
const store = new LazyStore("settings.json", { autoSave: true });

export async function getStored(key: string): Promise<string> {
    const path = await store.get<string>(key);
    return path ?? "";
}

export async function setStored(key: string, value: string): Promise<void> {
    await store.set(key, value);
}