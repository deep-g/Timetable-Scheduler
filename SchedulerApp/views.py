from django.http.response import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from collections import defaultdict
import random

POPULATION_SIZE = 30
NUMB_OF_ELITE_SCHEDULES = 2
TOURNAMENT_SELECTION_SIZE = 8
MUTATION_RATE = 0.05
VARS = {'generationNum': 0,
        'terminateGens': False}


class Population:
    def __init__(self, size):
        self._size = size
        self._data = data
        self._schedules = [Schedule().initialize() for i in range(size)]

    def getSchedules(self):
        return self._schedules


class Data:
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()
        self._sections = Section.objects.all()

    def get_rooms(self):
        return self._rooms

    def get_instructors(self):
        return self._instructors

    def get_courses(self):
        return self._courses

    def get_depts(self):
        return self._depts

    def get_meetingTimes(self):
        return self._meetingTimes

    def get_sections(self):
        return self._sections


class Class:
    def __init__(self, dept, section, course):
        self.department = dept
        self.course = course
        self.instructor = None
        self.room = None
        self.section = section
        self.valid_meeting_times = self.get_valid_meeting_times()
        self.meeting_time = random.choice(self.valid_meeting_times) if self.valid_meeting_times else None

    def get_id(self):
        return self.section_id

    def get_dept(self):
        return self.department

    def get_course(self):
        return self.course

    def get_instructor(self):
        return self.instructor

    def get_meetingTime(self):
        return self.meeting_time

    def get_room(self):
        return self.room

    def set_instructor(self, instructor):
        self.instructor = instructor

    def set_meetingTime(self, meetingTime):
        if isinstance(meetingTime, list):  # For lab courses, handle a list of consecutive slots
            if all(time in self.valid_meeting_times for time in meetingTime):
                self.meeting_time = meetingTime
            else:
                raise ValueError(f"Invalid meeting times: {meetingTime} for section {self.section}")
        elif meetingTime in self.valid_meeting_times:
            self.meeting_time = meetingTime
        else:
            raise ValueError(f"Invalid meeting time: {meetingTime} for section {self.section}")

    def set_room(self, room):
        self.room = room
    
    def get_valid_meeting_times(self):
        all_meeting_times = list(MeetingTime.objects.all())
        non_meeting_times = list(self.section.non_meeting_times.all())
        return [t for t in all_meeting_times if t not in non_meeting_times]

class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._isFitnessChanged = True

    def getClasses(self):
        self._isFitnessChanged = True
        return self._classes

    def getNumbOfConflicts(self):
        return self._numberOfConflicts

    def getFitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculateFitness()
            self._isFitnessChanged = False
        return self._fitness
    
    def addCourse(self, data, course, courses, dept, section):
        newClass = Class(dept, section, course)
        
        if not hasattr(self, "used_slots"):
            self.used_slots = {}  # Dictionary to track slots for each section

        if section not in self.used_slots:
            self.used_slots[section] = set()


        all_meeting_times = data.get_meetingTimes()
        non_meeting_times = list(section.non_meeting_times.all())
        valid_meeting_times = [t for t in all_meeting_times if t not in non_meeting_times]

        if course.is_lab:  # Handling lab courses separately
            consecutive_slots = None
            
            # Find two available consecutive slots that are not already used
            for i in range(len(valid_meeting_times) - 1):
                j = random.randint(0, len(valid_meeting_times) - 2)
                current_slot = valid_meeting_times[j]
                next_slot = valid_meeting_times[j + 1]

                if (
                    current_slot.day == next_slot.day and
                    int(next_slot.time.split(':')[0]) == int(current_slot.time.split(':')[0]) + 1 and
                    (current_slot.day, current_slot.time) not in self.used_slots[section] and
                    (next_slot.day, next_slot.time) not in self.used_slots[section]
                ):
                    consecutive_slots = [current_slot, next_slot]
                    # Mark slots as used
                    self.used_slots[section].add((current_slot.day, current_slot.time))
                    self.used_slots[section].add((next_slot.day, next_slot.time))
                    break

            if not consecutive_slots:
                self._numberOfConflicts += 1
                return

            # Assign the selected consecutive slots to the lab
            newClass.set_meetingTime(consecutive_slots)

            # Assign a lab room
            lab_rooms = [room for room in data.get_rooms() if room.room_type == 'lab']
            if lab_rooms:
                newClass.set_room(random.choice(lab_rooms))
            else:
                raise ValueError(f"No lab rooms available for lab course {course}")

        else:  # Handling lecture courses
            available_slots = [t for t in valid_meeting_times if (t.day, t.time) not in self.used_slots[section]]
            
            if available_slots:
                selected_time = random.choice(available_slots)
                newClass.set_meetingTime(selected_time)
                self.used_slots[section].add((selected_time.day, selected_time.time))
            else:
                raise ValueError(f"No valid meeting times available for section {section}")

            # Assign a lecture room
            lecture_rooms = [room for room in data.get_rooms() if room.room_type == 'lecture']
            if lecture_rooms:
                newClass.set_room(random.choice(lecture_rooms))
            else:
                raise ValueError(f"No lecture rooms available for course {course}")

        # Assign instructor
        crs_inst = course.instructors.all()
        selected_instructor = crs_inst[random.randrange(0, len(crs_inst))]
        newClass.set_instructor(selected_instructor)

        self._classes.append(newClass)
        return

    def initialize(self):
        sections = Section.objects.all()
        for section in sections:
            dept = section.department


            courses = dept.courses.all()
            for course in courses:
                for i in range(course.classes_per_week):
                    self.addCourse(data, course, courses, dept, section)


        return self

    def calculateFitness(self):
        self._numberOfConflicts = 0
        classes = self.getClasses()

        for i in range(len(classes)):
            # Seating capacity less them course student
            if classes[i].room.seating_capacity < int(classes[i].course.max_numb_students):
                self._numberOfConflicts += 1

            # print(classes[i].course.course_name, classes[i].meeting_time, classes[i].section, classes[i].room, classes[i].instructor)

            for j in range(i + 1, len(classes)):
                # Same course on same day
                if (classes[i].course.course_name == classes[j].course.course_name and \
                    str(classes[i].meeting_time).split()[1] == str(classes[j].meeting_time).split()[1]):
                    self._numberOfConflicts += 1

                # Teacher with lectures in different timetable at same time
                if (classes[i].section != classes[j].section and \
                    classes[i].meeting_time == classes[j].meeting_time and \
                    classes[i].instructor == classes[j].instructor):
                    self._numberOfConflicts += 1

                # Duplicate time in a department
                if (classes[i].section == classes[j].section and \
                    classes[i].meeting_time == classes[j].meeting_time):
                    self._numberOfConflicts += 1
                    
                # Lab subjects don't have 2 consecutive slots
                if classes[i].course.is_lab:
                    meeting_times = classes[i].meeting_time  # Assuming it's stored as a list

                    if meeting_times is None or len(meeting_times) != 2:
                        self._numberOfConflicts += 1  # Not exactly 2 slots assigned
                    else:
                        day1, time1 = meeting_times[0].day, int(meeting_times[0].time.split(':')[0])
                        day2, time2 = meeting_times[1].day, int(meeting_times[1].time.split(':')[0])

                        if day1 != day2 or time2 != time1 + 1:  # Not on the same day OR not consecutive
                            self._numberOfConflicts += 1

        return 1 / (self._numberOfConflicts + 1)


class GeneticAlgorithm:
    def evolve(self, population):
        return self._mutatePopulation(self._crossoverPopulation(population))

    def _crossoverPopulation(self, popula):
        crossoverPopula = Population(0)
        for i in range(NUMB_OF_ELITE_SCHEDULES):
            crossoverPopula.getSchedules().append(popula.getSchedules()[i])

        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            scheduleX = self._tournamentPopulation(popula)
            scheduleY = self._tournamentPopulation(popula)

            crossoverPopula.getSchedules().append(
                self._crossoverSchedule(scheduleX, scheduleY))

        return crossoverPopula

    def _mutatePopulation(self, population):
        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            self._mutateSchedule(population.getSchedules()[i])
        return population

    def _crossoverSchedule(self, scheduleX, scheduleY):
        crossoverSchedule = Schedule().initialize()
        for i in range(0, len(crossoverSchedule.getClasses())):
            if random.random() > 0.5:
                crossoverSchedule.getClasses()[i] = scheduleX.getClasses()[i]
            else:
                crossoverSchedule.getClasses()[i] = scheduleY.getClasses()[i]
        return crossoverSchedule

    def _mutateSchedule(self, mutateSchedule):
        schedule = Schedule().initialize()
        for i in range(len(mutateSchedule.getClasses())):
            if MUTATION_RATE > random.random():
                mutateSchedule.getClasses()[i] = schedule.getClasses()[i]
        return mutateSchedule

    def _tournamentPopulation(self, popula):
        tournamentPopula = Population(0)

        for i in range(0, TOURNAMENT_SELECTION_SIZE):
            tournamentPopula.getSchedules().append(
                popula.getSchedules()[random.randrange(0, POPULATION_SIZE)])

        # tournamentPopula.getSchedules().sort(key=lambda x: x.getFitness(),reverse=True)
        # return tournamentPopula
        return max(tournamentPopula.getSchedules(), key=lambda x: x.getFitness())



def context_manager(schedule):
    classes = schedule.getClasses()
    context = []
    for i in range(len(classes)):
        clas = {}
        clas['section'] = classes[i].section_id
        clas['dept'] = classes[i].department.dept_name
        clas['course'] = f'{classes[i].course.course_name} ({classes[i].course.course_number} {classes[i].course.max_numb_students})'
        clas['room'] = f'{classes[i].room.r_number} ({classes[i].room.seating_capacity})'
        clas['instructor'] = f'{classes[i].instructor.name} ({classes[i].instructor.uid})'
        clas['meeting_time'] = [
            classes[i].meeting_time.pid,
            classes[i].meeting_time.day,
            classes[i].meeting_time.time
        ]
        context.append(clas)
    return context


def apiGenNum(request):
    return JsonResponse({'genNum': VARS['generationNum']})

def apiterminateGens(request):
    VARS['terminateGens'] = True
    return redirect('home')



@login_required
def timetable(request):
    global data
    data = Data()
    population = Population(POPULATION_SIZE)
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False
    population.getSchedules().sort(key=lambda x: x.getFitness(), reverse=True)
    geneticAlgorithm = GeneticAlgorithm()
    schedule = population.getSchedules()[0]

    while (schedule.getFitness() != 1.0) and (VARS['generationNum'] < 10):
        if VARS['terminateGens']:
            return HttpResponse('')

        population = geneticAlgorithm.evolve(population)
        population.getSchedules().sort(key=lambda x: x.getFitness(), reverse=True)
        schedule = population.getSchedules()[0]
        VARS['generationNum'] += 1

        # for c in schedule.getClasses():
        #     print(c.course.course_name, c.meeting_time)
        print(f'\n> Generation #{VARS["generationNum"]}, Fitness: {schedule.getFitness()}')

    return render(
        request, 'timetable.html', {
            'schedule': schedule.getClasses(),
            'sections': data.get_sections(),
            'times': data.get_meetingTimes(),
            'timeSlots': TIME_SLOTS,
            'weekDays': DAYS_OF_WEEK
        })


'''
Page Views
'''

def home(request):
    return render(request, 'index.html', {})


@login_required
def instructorAdd(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('instructorAdd')
    context = {'form': form}
    return render(request, 'instructorAdd.html', context)


@login_required
def instructorEdit(request):
    context = {'instructors': Instructor.objects.all()}
    return render(request, 'instructorEdit.html', context)


@login_required
def instructorDelete(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('instructorEdit')


@login_required
def roomAdd(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('roomAdd')
    context = {'form': form}
    return render(request, 'roomAdd.html', context)


@login_required
def roomEdit(request):
    context = {'rooms': Room.objects.all()}
    return render(request, 'roomEdit.html', context)


@login_required
def roomDelete(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('roomEdit')


@login_required
def meetingTimeAdd(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('meetingTimeAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'meetingTimeAdd.html', context)


@login_required
def meetingTimeEdit(request):
    context = {'meeting_times': MeetingTime.objects.all()}
    return render(request, 'meetingTimeEdit.html', context)


@login_required
def meetingTimeDelete(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('meetingTimeEdit')


@login_required
def courseAdd(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('courseAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'courseAdd.html', context)


@login_required
def courseEdit(request):
    instructor = defaultdict(list)
    course_instructors = Course.instructors.through.objects.select_related('course', 'instructor')

    for course_relation in course_instructors:
        course_obj = course_relation.course  # Access the Course object
        instructor_name = course_relation.instructor.name  # Access Instructor object
        instructor[course_obj.course_number].append(instructor_name)  # Append the instructor name

    context = {'courses': Course.objects.all(), 'instructor': instructor}
    return render(request, 'courseEdit.html', context)


@login_required
def courseDelete(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('courseEdit')


@login_required
def departmentAdd(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('departmentAdd')
    context = {'form': form}
    return render(request, 'departmentAdd.html', context)


@login_required
def departmentEdit(request):
    course = defaultdict(list)
    for dept in Department.courses.through.objects.all():
        dept_name = Department.objects.filter(
            id=dept.department_id).values('dept_name')[0]['dept_name']
        course_name = Course.objects.filter(
            course_number=dept.course_id).values(
                'course_name')[0]['course_name']
        course[dept_name].append(course_name)

    context = {'departments': Department.objects.all(), 'course': course}
    return render(request, 'departmentEdit.html', context)


@login_required
def departmentDelete(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('departmentEdit')


@login_required
def sectionAdd(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('sectionAdd')
    context = {'form': form}
    return render(request, 'sectionAdd.html', context)


@login_required
def sectionEdit(request, pk=None):
    if pk:
        section = get_object_or_404(Section, pk=pk)
        form = SectionForm(request.POST or None, instance=section)
        if request.method == 'POST':
            if form.is_valid():
                form.save()
                return redirect('sectionEdit', pk=pk)
        context = {'form': form, 'section': section}
        return render(request, 'sectionEditForm.html', context)
    else:
        context = {'sections': Section.objects.all()}
        return render(request, 'sectionEdit.html', context)


@login_required
def sectionDelete(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('sectionEdit')




'''
Error pages
'''

def error_404(request, exception):
    return render(request,'errors/404.html', {})

def error_500(request, *args, **argv):
    return render(request,'errors/500.html', {})
