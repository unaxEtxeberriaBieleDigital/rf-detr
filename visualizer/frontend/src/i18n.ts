import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import es_ES from './locales/es_ES.json';
import en_US from './locales/en_US.json';

const resources = {
    en_US: {
        translation: en_US
    },
    es_ES: {
        translation: es_ES
    }
};

i18n
    .use(LanguageDetector) // detecta el idioma del sistema/navegador
    .use(initReactI18next) // pasa la instancia a react-i18next
    .init({
        resources,
        fallbackLng: 'en_US',
        interpolation: {
            escapeValue: false // react ya protege contra XSS por defecto
        }
    });

if (import.meta.hot) {
    import.meta.hot.accept(['./locales/es_ES.json', './locales/en_US.json'], (modules) => {
        const [newEs, newEn] = modules;

        if (newEs) {
            const dataEs = newEs.default ?? newEs;
            i18n.addResourceBundle('es_ES', 'translation', dataEs, true, true);
        }

        if (newEn) {
            const dataEn = newEn.default ?? newEn;
            i18n.addResourceBundle('en_US', 'translation', dataEn, true, true);
        }

        // Fuerza a react-i18next a re-evaluar el idioma actual y re-renderizar los componentes
        i18n.changeLanguage(i18n.language);
    });
}

export default i18n;