-- Pandoc filter for `make docx`.
-- Team notes (\todo, \decide, and the ideas environment) are red in the PDF and
-- stay red in Word through the "Team Note" character style defined in
-- tools/reference.docx; everything else is black.
-- \paragraph{} headings stay unnumbered, as in the PDF.

local function team(inlines)
  return pandoc.Span(inlines, {['custom-style'] = 'Team Note'})
end

function Span(el)
  local style = el.attributes.style or ''
  if style:match('color:%s*red') then
    return team(el.content)
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
