from django import template
register = template.Library()

@register.filter
def dictKey(d, k):
    '''Returns the given key from a dictionary.'''

    return ', '.join(d[k]) or ''


@register.simple_tag
def sub(s, d, w, t):
    '''Returns the subject-teacher for a department, weekday and time period'''
    for c in s:
        if c.department.dept_name == d:
            # Check if meeting_time is a list (for labs)
            if isinstance(c.meeting_time, list):
                # Iterate over all meeting times
                for i, mt in enumerate(c.meeting_time):
                    if mt.day == w and mt.time == t:
                        # Ensure lab occupies multiple slots by checking the next meeting time
                        if i + 1 < len(c.meeting_time):  
                            return f'{c.course.course_name} (Lab) ({c.instructor.name}) - Room: {c.room.r_number} (Continued)'
                        return f'{c.course.course_name} (Lab) ({c.instructor.name}) - Room: {c.room.r_number}'
            else:  # Regular single-hour lecture
                if c.meeting_time.day == w and c.meeting_time.time == t:
                    return f'{c.course.course_name} ({c.instructor.name}) - Room: {c.room.r_number}'
    return ''

@register.tag
def active(parser, token):
    args = token.split_contents()
    template_tag = args[0]
    if len(args) < 2:
        raise (template.TemplateSyntaxError, f'{template_tag} tag requires at least one argument')
    return NavSelectedNode(args[1:])

class NavSelectedNode(template.Node):
    def __init__(self, patterns):
        self.patterns = patterns
    def render(self, context):
        path = context['request'].path
        for p in self.patterns:
            pValue = template.Variable(p).resolve(context)
            if path == pValue:
                return 'active'
        return ''