# shared/

Code used by both **mobile production** and **web CV lab**.

- `config/flags.ts` — `WEB_DEBUG`, `MOBILE_PRODUCTION`, `IS_NATIVE`
- `config/index.ts` — re-exports flags + runtime config

Do not import `@/debug/web` from mobile or vision production paths.
