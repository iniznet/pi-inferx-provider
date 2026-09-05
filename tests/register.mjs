// Node >= 22.15 installs hooks in-thread; older releases need a worker module path.
import * as nodeModule from "node:module";
import { hooks } from "./loader.mjs";

if (typeof nodeModule.registerHooks === "function") {
  nodeModule.registerHooks(hooks);
} else {
  nodeModule.register("./loader.mjs", import.meta.url);
}
