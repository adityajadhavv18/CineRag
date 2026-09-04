# Screenshots

Drop the files below into this folder using these **exact** filenames and the README
picks them up automatically. Any slot you skip just renders as a broken image — delete
that block from `../../README.md` if you don't want it.

| Filename | What to capture | Where |
| --- | --- | --- |
| `hero.png` | The full browse page — hero carousel plus a couple of genre rows. This is the one at the very top, so make it the best-looking shot. | `localhost:5173` |
| `browse-hero.png` | Just the hero carousel | `localhost:5173` |
| `browse-rows.png` | Genre rows, ideally with one card hovered to show the blurb + tags preview | `localhost:5173` |
| `chat-answer.png` | The chat drawer with a finished, cited answer — `[1]`-style markers visible | ask *"gritty crime dramas starring Denzel Washington"* |
| `chat-results.png` | The shelf after an answer, i.e. genre rows replaced by the cited films | same query, scroll behind the drawer |
| `clarification.png` | The clarify card with its counted options as buttons | ask *"recommend some action movies"* |
| `movie-detail.png` | The detail modal — cast, "similar", and a franchise timeline | open a film in a collection, e.g. TMDB id `671` (Harry Potter) |
| `person-detail.png` | A person's filmography, acted + directed | click a director's name in a detail modal |
| `neo4j-graph.png` | Neo4j Browser with a rendered subgraph | `localhost:7474` — try `MATCH p=(:Person)-[:DIRECTED]->(:Movie)-[:HAS_GENRE]->(:Genre) RETURN p LIMIT 40` |
| `qdrant-dashboard.png` | The `movies` collection, showing point count and vector config | `localhost:6333/dashboard` |
| `langsmith-trace.png` | One agent run expanded, showing the node path | LangSmith, if tracing is on |

## Capture tips

- Browser window **1440px wide or wider**; 2× device pixel ratio if your display supports it.
- Hide bookmark bars, extensions and any personal tabs — crop to page content only.
- PNG for UI. Keep each file **under ~1.5 MB** so the README loads quickly; run them
  through `pngquant` or ImageOptim if they're heavy.
- For the side-by-side pairs, capture both at the same width so the table columns line up.

On macOS: `Cmd+Shift+4`, then `Space` to grab a whole window, or hold `Option` while
releasing to drop the window shadow.
