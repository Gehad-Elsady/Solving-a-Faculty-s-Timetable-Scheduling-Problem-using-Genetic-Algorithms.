import prettytable as prettytable
import random as rnd
import numpy as np

POPULATION_SIZE = 9
NUMB_OF_ELITE_SCHEDULES = 1
TOURNAMENT_SELECTION_SIZE = 3
MUTATION_RATE = 0.1


class Data:
    print("enter the time of LUC as From 9:00 AM  to   11:00 AM")
    v1 = input("LUC1")
    v2 = input("LUC2")
    v3 = input("LUC3")
    v4 = input("LUC4")
    print("enter DR name as DR/ AMR S. GHONIEM ")
    v5 = input("DR name1")
    v6 = input("DR name2")
    v7 = input("DR name3")
    v8 = input("DR name4")
    ROOMS = [["H1", 400], ["H2", 350], ["H3", 200], ["H4", 150]]
    MEETING_TIMES = [["LEC1", v1], ["LEC2", v2],
                     ["LEC3", v3], ["LEC4", v4]]

    INSTRUCTORS = [["I1", v5], ["I2", v6],
                   ["I3", v7], ["I4", v8]]

    def __init__(self):
        self._rooms = []
        self._meetingTimes = []
        self._instructors = []
        for i in range(0, len(self.ROOMS)):
            self._rooms.append(Room(self.ROOMS[i][0], self.ROOMS[i][1]))
        for i in range(0, len(self.MEETING_TIMES)):
            self._meetingTimes.append(MeetingTime(self.MEETING_TIMES[i][0], self.MEETING_TIMES[i][1]))
        for i in range(0, len(self.INSTRUCTORS)):
            self._instructors.append(Instructor(self.INSTRUCTORS[i][0], self.INSTRUCTORS[i][1]))
        course1 = Course("C1", "ST-121", [self._instructors[0], self._instructors[1]], 120)
        course2 = Course("C2", "CS-214", [self._instructors[0], self._instructors[1], self._instructors[2]], 150)
        course3 = Course("C3", "CS-314", [self._instructors[0], self._instructors[1]], 200)
        course4 = Course("C4", "IS-241", [self._instructors[2], self._instructors[3]], 100)
        course5 = Course("C5", "IT-222", [self._instructors[3]], 140)
        course6 = Course("C6", "IS-211", [self._instructors[0], self._instructors[2]], 190)
        course7 = Course("C7", "IS-444", [self._instructors[1], self._instructors[3]], 120)
        self._courses = [course1, course2, course3, course4, course5, course6, course7]
        dept1 = Department("CS", [course1, course3])
        dept2 = Department("IS", [course2, course4, course5])
        dept3 = Department("IT", [course6, course7])
        self._depts = [dept1, dept2, dept3]
        self._numberOfClasses = 0
        for i in range(0, len(self._depts)):
            self._numberOfClasses += len(self._depts[i].get_courses())

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

    def get_numberOfClasses(self):
        return self._numberOfClasses


class Differential_Algorithm :
    def _init_(self,
               population,
               fitness_vec_size,
               l_bound,
               u_bound,
               f,
               evolve,
               cross_prob):

        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population = population
        self.fitness_vector_size = fitness_vec_size
        self.population = None
        self.mutants = list()
        self.trial = list()
        self.f = f
        self.cr = cross_prob
        self.evolve = evolve

    def population(self):
        rand_vals = (self.u_bound - self.l_bound) * \
                    np.random.random((self.population_size, self.fitness_vector_size)) + self.l_bound
        self.population = np.array([Individual(p) for p in rand_vals])


    def mutate(self):
        for i, val in enumerate(self.population):
            indexes = np.random.permutation(np.delete(np.arange(len(self.population)), i))[:3]
            a, b, c = [x.phenotypes for x in self.population[indexes]]
            use_f = self.f or np.random.random() / 2.0 + 0.5
            self.mutants.append(a + use_f * (b - c))

    def crossover(self):
        for v, x in zip(self.mutants, [x.phenotypes for x in self.population]):
            tmp_vec = list()
            for (j, (vi, xi)) in enumerate(zip(v, x)):
                tmp_vec.append(vi if np.random.random() <= self.cr or
                                     j == np.random.randint(0, self.fitness_vector_size) else xi)
            self.trial.append(Individual(tmp_vec))

    def select(self):
        self.population = np.array([x if x.fitness < u.fitness else u for x, u in zip(self.population, self.trial)])
        self.trial.clear()
        self.mutants.clear()


def fitness_func(phenotypes):
    pass


class Individual(object):
    def _init_(self, phenotypes):
        self.phenotypes = np.array(phenotypes)  # phenotype
        self.fitness = fitness_func(self.phenotypes)  # value of the fitness function

    def _str_(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)



    def fitness(x, y):
        # Funcion Rosenbrock en 2D
        return 100 * ((y - (x ** 2)) ** 2) + ((1 - (x ** 2)) ** 2)


class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numbOfConflicts = 0
        self._fitness = -1
        self._classNumb = 0
        self._isFitnessChanged = True
    def get_classes(self):
        self._isFitnessChanged = True
        return self._classes
    def get_numbOfConflicts(self): return self._numbOfConflicts
    def get_fitness(self):
        if (self._isFitnessChanged == True):
            self._fitness = self.calculate_fitness()
            self._isFitnessChanged = False
        return self._fitness
    def initialize(self):
        depts = self._data.get_depts()
        for i in range(0, len(depts)):
            courses = depts[i].get_courses()
            for j in range(0, len(courses)):
                newClass = Class(self._classNumb, depts[i], courses[j])
                self._classNumb += 1
                newClass.set_meetingTime(data.get_meetingTimes()[rnd.randrange(0, len(data.get_meetingTimes()))])
                newClass.set_room(data.get_rooms()[rnd.randrange(0, len(data.get_rooms()))])
                newClass.set_instructor(courses[j].get_instructors()[rnd.randrange(0, len(courses[j].get_instructors()))])
                self._classes.append(newClass)
        return self
    def calculate_fitness(self):
        self._numbOfConflicts = 0
        classes = self.get_classes()
        for i in range(0, len(classes)):
            if (classes[i].get_room().get_seatingCapacity() < classes[i].get_course().get_maxNumbOfStudents()):
                self._numbOfConflicts += 1
            for j in range(0, len(classes)):
                if (j >= i):
                    if (classes[i].get_meetingTime() == classes[j].get_meetingTime() and
                    classes[i].get_id() != classes[j].get_id()):
                        if (classes[i].get_room() == classes[j].get_room()): self._numbOfConflicts += 1
                        if (classes[i].get_instructor() == classes[j].get_instructor()): self._numbOfConflicts += 1
        return 1 / ((1.0*self._numbOfConflicts + 1))
    def __str__(self):
        returnValue = ""
        for i in range(0, len(self._classes)-1):
            returnValue += str(self._classes[i]) + ", "
        returnValue += str(self._classes[len(self._classes)-1])
        return returnValue

class Population:
    def __init__(self, size):
        self._size = size
        self._data = data
        self._schedules = []
        for i in range(0, size): self._schedules.append(Schedule().initialize())
    def get_schedules(self): return self._schedules


def main(max_epochs=None):
    de= Differential_Algorithm()

    de.generate_population()
    print('Initial population')
    for ind in sorted(de.population, key=lambda x: x.fitness):
        print(ind)
    for i in range(max_epochs):
        de.mutate()
        de.crossover()
        de.select()
        print('{0}/{1} Current population:'.format(i + 1, max_epochs))
        print(sorted(de.population, key=lambda x: x.fitness)[0])


class Course:
    def __init__(self, number, name, instructors, maxNumbOfStudents):
        self._number = number
        self._name = name
        self._maxNumbOfStudents = maxNumbOfStudents
        self._instructors = instructors

    def get_number(self): return self._number

    def get_name(self): return self._name

    def get_instructors(self): return self._instructors

    def get_maxNumbOfStudents(self): return self._maxNumbOfStudents

    def __str__(self): return self._name


class Instructor:
    def __init__(self, id, name):
        self._id = id
        self._name = name

    def get_id(self): return self._id

    def get_name(self): return self._name

    def __str__(self): return self._name


class Room:
    def __init__(self, number, seatingCapacity):
        self._number = number
        self._seatingCapacity = seatingCapacity

    def get_number(self): return self._number

    def get_seatingCapacity(self): return self._seatingCapacity


class MeetingTime:
    def __init__(self, id, time):
        self._id = id
        self._time = time

    def get_id(self): return self._id

    def get_time(self): return self._time


class Department:
    def __init__(self, name, courses):
        self._name = name
        self._courses = courses

    def get_name(self): return self._name

    def get_courses(self): return self._courses


class Class:
    def __init__(self, id, dept, course):
        self._id = id
        self._dept = dept
        self._course = course
        self._instructor = None
        self._meetingTime = None
        self._room = None

    def get_id(self): return self._id

    def get_dept(self): return self._dept

    def get_course(self): return self._course

    def get_instructor(self): return self._instructor

    def get_meetingTime(self): return self._meetingTime

    def get_room(self): return self._room

    def set_instructor(self, instructor): self._instructor = instructor

    def set_meetingTime(self, meetingTime): self._meetingTime = meetingTime

    def set_room(self, room): self._room = room

    def __str__(self):
        return str(self._dept.get_name()) + "," + str(self._course.get_number()) + "," + \
               str(self._room.get_number()) + "," + str(self._instructor.get_id()) + "," + str(
            self._meetingTime.get_id())


class DisplayMgr:
    def print_available_data(self):
        print("> All Available Data")
        self.print_dept()
        self.print_course()
        self.print_room()
        self.print_instructor()
        self.print_meeting_times()

    def print_dept(self):
        depts = data.get_depts()
        availableDeptsTable = prettytable.PrettyTable(['dept', 'courses'])
        for i in range(0, len(depts)):
            courses = depts.__getitem__(i).get_courses()
            tempStr = "["
            for j in range(0, len(courses) - 1):
                tempStr += courses[j].__str__() + ", "
            tempStr += courses[len(courses) - 1].__str__() + "]"
            availableDeptsTable.add_row([depts.__getitem__(i).get_name(), tempStr])
        print(availableDeptsTable)

    def print_course(self):
        availableCoursesTable = prettytable.PrettyTable(['id', 'course #', 'max # of students', 'instructors'])
        courses = data.get_courses()
        for i in range(0, len(courses)):
            instructors = courses[i].get_instructors()
            tempStr = ""
            for j in range(0, len(instructors) - 1):
                tempStr += instructors[j].__str__() + ", "
            tempStr += instructors[len(instructors) - 1].__str__()
            availableCoursesTable.add_row(
                [courses[i].get_number(), courses[i].get_name(), str(courses[i].get_maxNumbOfStudents()), tempStr])
        print(availableCoursesTable)

    def print_instructor(self):
        availableInstructorsTable = prettytable.PrettyTable(['id', 'instructor'])
        instructors = data.get_instructors()
        for i in range(0, len(instructors)):
            availableInstructorsTable.add_row([instructors[i].get_id(), instructors[i].get_name()])
        print(availableInstructorsTable)

    def print_room(self):
        availableRoomsTable = prettytable.PrettyTable(['room #', 'max seating capacity'])
        rooms = data.get_rooms()
        for i in range(0, len(rooms)):
            availableRoomsTable.add_row([str(rooms[i].get_number()), str(rooms[i].get_seatingCapacity())])
        print(availableRoomsTable)

    def print_meeting_times(self):
        availableMeetingTimeTable = prettytable.PrettyTable(['id', 'Meeting Time'])
        meetingTimes = data.get_meetingTimes()
        for i in range(0, len(meetingTimes)):
            availableMeetingTimeTable.add_row([meetingTimes[i].get_id(), meetingTimes[i].get_time()])
        print(availableMeetingTimeTable)

    def print_generation(self, population):
        table1 = prettytable.PrettyTable(
            ['schedule #', 'fitness', '# of conflicts', 'classes [dept,class,room,instructor,meeting-time]'])
        schedules = population.get_schedules()
        for i in range(0, len(schedules)):
            table1.add_row([str(i), round(schedules[i].get_fitness(), 3), schedules[i].get_numbOfConflicts(),
                            schedules[i].__str__()])
        print(table1)

    def print_schedule_as_table(self, schedule):
        classes = schedule.get_classes()
        table = prettytable.PrettyTable(
            ['Class #', 'Dept', 'Course (number, max # of students)', 'Room (Capacity)', 'Instructor (Id)',
             'Meeting Time (Id)'])
        for i in range(0, len(classes)):
            table.add_row([str(i), classes[i].get_dept().get_name(), classes[i].get_course().get_name() + " (" +
                           classes[i].get_course().get_number() + ", " +
                           str(classes[i].get_course().get_maxNumbOfStudents()) + ")",
                           classes[i].get_room().get_number() + " (" + str(
                               classes[i].get_room().get_seatingCapacity()) + ")",
                           classes[i].get_instructor().get_name() + " (" + str(
                               classes[i].get_instructor().get_id()) + ")",
                           classes[i].get_meetingTime().get_time() + " (" + str(
                               classes[i].get_meetingTime().get_id()) + ")"])
        print(table)


data = Data()
displayMgr = DisplayMgr()
displayMgr.print_available_data()
generationNumber = 0
print("\n> best solution "+str(generationNumber))
population = Population(POPULATION_SIZE)
population.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
displayMgr.print_generation(population)
displayMgr.print_schedule_as_table(population.get_schedules()[0])
de = Differential_Algorithm()
while (population.get_schedules()[0].get_fitness() != 1.0):
    generationNumber += 1
    print("\n> Generation # " + str(generationNumber))
    population = de.evolve(population)
    population.get_schedules().sort(key=lambda x: x.get_fitness(), reverse=True)
    displayMgr.print_generation(population)
    displayMgr.print_schedule_as_table(population.get_schedules()[0])
print("\n\n")

