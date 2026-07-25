#!/usr/bin/env node
// PreToolUse hook: blocks git commit/push/merge against main or master, and any force push.
// This repo is PR-only — the backend developer merges, Claude never does.
const { execSync } = require("child_process");

const PROTECTED = new Set(["main", "master"]);

function currentBranch() {
  try {
    return execSync("git branch --show-current", { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function block(reason) {
  process.stderr.write(reason + "\n");
  process.exit(2);
}

let raw = "";
process.stdin.on("data", (chunk) => (raw += chunk));
process.stdin.on("end", () => {
  let command = "";
  try {
    command = JSON.parse(raw).tool_input?.command || "";
  } catch {
    process.exit(0);
  }

  if (/push/.test(command) && (/--force/.test(command) || /(^|\s)-f(\s|$)/.test(command))) {
    block(`Blocked: '${command}' is a force push, which is never allowed in this repo.`);
  }

  if (/git\s+(commit|push|merge)/.test(command)) {
    const branch = currentBranch();
    if (branch && PROTECTED.has(branch)) {
      block(
        `Blocked: '${command}' would affect protected branch '${branch}'. ` +
        "This repo is PR-only — create a feature branch, then open a PR for the " +
        "backend developer to review and merge."
      );
    }
  }

  process.exit(0);
});
