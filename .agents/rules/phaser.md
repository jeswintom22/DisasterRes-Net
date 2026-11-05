# Phaser Game Engine Rules

Rules for all code in `app/frontend/src/game/`. These exist because we've been burned by each one.

## Cleanup & Memory

1. **Every game object created in `create()` must be destroyed in `shutdown()`.** Store references in instance fields — never create fire-and-forget graphics/text/zones. Helper functions that create objects must return them so the caller can track and destroy them.
2. **Every `gameStore.on()` call must store its unsubscribe function** in `this.storeUnsubscribers[]` and call them all in `shutdown()`.
3. **Every DOM `addEventListener` on the canvas must have a matching `removeEventListener` in `shutdown()`.** Store the handler reference.
4. **Register `shutdown()` on Phaser's lifecycle** via `this.events.once('shutdown', ...)` in the constructor.
5. **Call `this.tweens.killAll()` before destroying game objects** in `shutdown()`.
6. **Never use infinite tweens (`repeat: -1`) without a cleanup path.**

## Performance

7. **Use squared distance** (`dx*dx + dy*dy < threshold*threshold`) — never `Math.sqrt` in per-frame code.
8. **Frame-rate independent lerp**: use `1 - Math.exp(-speed * delta/1000)`, never a fixed factor per frame.
9. **Avoid creating/destroying game objects in `update()`.** Pool or toggle visibility instead.
10. **Cache computed string keys** in a Map if used in hot paths.

## Architecture

11. **Don't grow God Scenes.** Extract new features into manager classes rather than adding methods to the scene.
12. **All React ↔ Phaser communication goes through `OfficeBridge`.** Never import React stores directly in scene code (exception: `usePersonStore.getState()` in `preload()` for initial data load only).
13. **Helper functions that create Phaser objects must return them** so the scene can track and destroy them.
