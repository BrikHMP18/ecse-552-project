-- Pandoc filter for `make docx`.
-- Team notes (\todo, \decide, and the ideas environment) are red in the PDF and
-- stay red in Word through the "Team Note" character style defined in
-- docs/tools/reference.docx; everything else is black.
-- \paragraph{} titles run into their paragraph in bold, as in the PDF.

local function team(inlines)
  return pandoc.Span(inlines, {['custom-style'] = 'Team Note'})
end

function Span(el)
  local style = el.attributes.style or ''
  if style:match('color:%s*red') then
    return team(el.content)
  elseif style:match('color:%s*review') then
    -- \review{}: text the team still has to check, yellow highlight.
    return pandoc.Span(el.content, {['custom-style'] = 'Team Review'})
  elseif style:match('color:%s*teamlight') then
    -- \member{}: names to fill in, light red.
    return pandoc.Span(el.content, {['custom-style'] = 'Team Name'})
  end
end

function Div(el)
  if el.classes:includes('ideas') then
    return pandoc.walk_block(el, {
      Plain = function(p) return pandoc.Plain({team(p.content)}) end,
      Para = function(p) return pandoc.Para({team(p.content)}) end,
    }).content
  end
end

function Header(el)
  if el.level >= 4 then
    el.classes:insert('unnumbered')
    return el
  end
end

-- Tables: pandoc drops LaTeX column widths and reads the first row as body.
-- Take the widths from the p{...} columns of each tabular in the source (in
-- order), and make the header row bold.
local LCOL_CM = 1.0 -- width given to l/c/r columns, which have no explicit width

local function tabular_widths()
  local specs = {}
  local f = io.open(PANDOC_STATE.input_files[1])
  if not f then return specs end
  local src = f:read('a')
  f:close()
  for spec in src:gmatch('\\begin{tabular}(%b{})') do
    spec = spec:sub(2, -2):gsub('[@>]%b{}', '')
    local cols = {}
    local i = 1
    while i <= #spec do
      local ch = spec:sub(i, i)
      if ch == 'p' or ch == 'm' or ch == 'b' then
        local arg = spec:match('^%b{}', i + 1)
        table.insert(cols, tonumber(arg:match('([%d.]+)cm')) or LCOL_CM)
        i = i + 1 + #arg
      else
        if ch == 'l' or ch == 'c' or ch == 'r' then table.insert(cols, LCOL_CM) end
        i = i + 1
      end
    end
    table.insert(specs, cols)
  end
  return specs
end

local widths = nil
local table_index = 0

function Table(tbl)
  widths = widths or tabular_widths()
  table_index = table_index + 1
  local cols = widths[table_index]
  if cols and #cols == #tbl.colspecs then
    local total = 0
    for _, w in ipairs(cols) do total = total + w end
    for k, w in ipairs(cols) do tbl.colspecs[k] = {tbl.colspecs[k][1], w / total} end
  end
  -- The row above \midrule is the header; make it bold.
  local body = tbl.bodies[1]
  if #tbl.head.rows == 0 and body and #body.body > 0 then
    tbl.head.rows = {table.remove(body.body, 1)}
  end
  for _, row in ipairs(tbl.head.rows) do
    for _, cell in ipairs(row.cells) do
      for k, block in ipairs(cell.contents) do
        if block.t == 'Plain' or block.t == 'Para' then
          block.content = {pandoc.Strong(block.content)}
          cell.contents[k] = block
        end
      end
    end
  end
  return tbl
end

-- \paragraph{} titles: run them into the following paragraph in bold, as the
-- NeurIPS PDF does, instead of a heading line of their own (saves a line each).
function Blocks(blocks)
  local out = pandoc.Blocks{}
  local i = 1
  while i <= #blocks do
    local b, nxt = blocks[i], blocks[i + 1]
    if b.t == 'Header' and b.level >= 4 and nxt and (nxt.t == 'Para' or nxt.t == 'Plain') then
      local inlines = pandoc.Inlines{pandoc.Strong(b.content), pandoc.Space()}
      inlines:extend(nxt.content)
      out:insert(pandoc.Para(inlines))
      i = i + 2
    else
      out:insert(b)
      i = i + 1
    end
  end
  return out
end
