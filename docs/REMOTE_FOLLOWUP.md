# Optional GitHub follow-up — approval required

These are prepared suggestions, not changes made by this polish pass. Normal
source commits and CI use the existing repository; topics, tags, Releases,
visibility, and other repository settings remain unchanged.

## Suggested topics

`pytorch`, `computer-vision`, `object-detection`, `fastapi`,
`industrial-inspection`, `model-serving`

After approval, use the repository's **About → Edit → Topics** field, or:

```powershell
gh repo edit ThePhysic/visual-inspection-dl --add-topic pytorch --add-topic computer-vision --add-topic object-detection --add-topic fastapi --add-topic industrial-inspection --add-topic model-serving
```

The current About description already describes the component accurately. No
description or visibility change is proposed.

## Suggested model tag and Release

- Tag: `pcb-ind-ssdlite-v1`
- Title: `PCB-IND SSDLite v1 — research and portfolio snapshot`
- Target: the exact reviewed polish commit that the owner approves; record its
  full SHA before executing any commands below.
- Notes: [prepared Release notes](RELEASE_NOTES_DRAFT.md).
- Assets: none. The [existing in-repository bundle](../releases/pcb-ind-ssdlite-v1/)
  already contains the weights, metadata and evaluation. Do not upload a duplicate.

Only after separate authorization, check for existing tags/Releases, replace the
placeholder with the approved SHA, inspect it, and create/push the annotated tag:

```powershell
gh release list --repo ThePhysic/visual-inspection-dl
git tag --list pcb-ind-ssdlite-v1
$approvedCommit = 'REPLACE_WITH_APPROVED_COMMIT_SHA'
git show --no-patch --oneline $approvedCommit
git tag -a pcb-ind-ssdlite-v1 $approvedCommit -m "PCB-IND SSDLite v1 research snapshot"
git push origin refs/tags/pcb-ind-ssdlite-v1
gh release create pcb-ind-ssdlite-v1 --repo ThePhysic/visual-inspection-dl --verify-tag --title "PCB-IND SSDLite v1 — research and portfolio snapshot" --notes-file docs/RELEASE_NOTES_DRAFT.md
```

`--verify-tag` requires the remote tag to exist. If a tag or Release with that
name already exists, stop and inspect it rather than replacing it. No force
push or history rewrite is part of these instructions.

The equivalent GitHub UI is **Releases → Draft a new release**: select the
approved tag, paste the prepared notes, leave binary assets empty, review the
license and measurement qualifiers, then publish only with authorization.
