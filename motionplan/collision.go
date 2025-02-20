package motionplan

import (
	"math"

	spatial "go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/utils"
)

// Collision is a pair of strings corresponding to names of Geometry objects in collision, and a penetrationDepth describing the Euclidean
// distance a Geometry would have to be moved to resolve the Collision.
type Collision struct {
	name1, name2     string
	penetrationDepth float64
}

// collisionsAlmostEqual compares two Collisions and returns if they are almost equal.
func collisionsAlmostEqual(c1, c2 Collision) bool {
	return ((c1.name1 == c2.name1 && c1.name2 == c2.name2) || (c1.name1 == c2.name2 && c1.name2 == c2.name1)) &&
		utils.Float64AlmostEqual(c1.penetrationDepth, c2.penetrationDepth, 0.1)
}

// collisionListsAlmostEqual compares two lists of Collisions and returns if they are almost equal.
func collisionListsAlmostEqual(cs1, cs2 []Collision) bool {
	if len(cs1) != len(cs2) {
		return false
	}

	// loop through list 1 and match with elements in list 2, mark on list of used indexes
	used := make([]bool, len(cs1))
	for _, c1 := range cs1 {
		for i, c2 := range cs2 {
			if collisionsAlmostEqual(c1, c2) {
				used[i] = true
				break
			}
		}
	}

	// loop through list of used indexes
	for _, c := range used {
		if !c {
			return false
		}
	}
	return true
}

// geometryGraph is a struct that stores distance relationships between sets of geometries.
type geometryGraph struct {
	// x and y are the two sets of geometries, each of which will be compared to the geometries in the other set
	x, y map[string]spatial.Geometry

	// distances is the data structure to store the distance relationships between two named geometries
	// can be acessed as distances[name1][name2] to get the distance between name1 and name2
	distances map[string]map[string]float64
}

// newGeometryGraph instantiates a geometryGraph with the x and y geometry sets.
func newGeometryGraph(x, y map[string]spatial.Geometry) geometryGraph {
	distances := make(map[string]map[string]float64)
	for name := range x {
		distances[name] = make(map[string]float64)
	}
	return geometryGraph{
		x:         x,
		y:         y,
		distances: distances,
	}
}

// setDistance takes two given geometry names and sets their distance in the distances table exactly once
// since the relationship between the geometries is bidirectional, the order that the names are passed in is not important.
func (gg *geometryGraph) setDistance(xName, yName string, distance float64) {
	if _, ok := gg.distances[yName][xName]; ok {
		gg.distances[yName][xName] = distance
	} else {
		gg.distances[xName][yName] = distance
	}
}

// getDistance finds the distance between the given geometry names by referencing the distances table
// a secondary return value of type bool is also returned, indicating if the distance was found in the table
// if the distance between the geometry names was never set, the return value will be (NaN, false).
func (cg *collisionGraph) getDistance(name1, name2 string) (float64, bool) {
	if distance, ok := cg.distances[name1][name2]; ok {
		return distance, true
	}
	if distance, ok := cg.distances[name2][name1]; ok {
		return distance, true
	}
	return math.NaN(), false
}

// collisionGraph utilizes the geometryGraph structure to make collision checks between geometries
// a collision is defined as a negative penetration depth and is stored in the distances table.
type collisionGraph struct {
	geometryGraph

	// reportDistances is a bool that determines how the collisionGraph will report collisions
	//    - true:  all distances will be determined and numerically reported
	//    - false: collisions will be reported as bools, not numerically. Upon finding a collision, will exit early
	reportDistances bool
}

// newCollisionGraph instantiates a collisionGraph object and checks for collisions between the x and y sets of geometries
// collisions that are reported in the reference CollisionSystem argument will be ignored and not stored as edges in the graph.
// if the set y is nil, the graph will be instantiated with y = x.
func newCollisionGraph(x, y map[string]spatial.Geometry, reference *collisionGraph, reportDistances bool) (cg *collisionGraph, err error) {
	if y == nil {
		y = x
	}
	cg = &collisionGraph{
		geometryGraph:   newGeometryGraph(x, y),
		reportDistances: reportDistances,
	}

	var distance float64
	for xName, xGeometry := range cg.x {
		for yName, yGeometry := range cg.y {
			if _, ok := cg.getDistance(xName, yName); ok || xGeometry == yGeometry {
				// geometry pair already has distance information associated with it, or is comparing with itself - skip to next pair
				continue
			}
			if reference != nil && reference.collisionBetween(xName, yName) {
				// represent previously seen collisions as NaNs
				// per IEE standards, any comparison with NaN will return false, so these will never be considered collisions
				distance = math.NaN()
			} else if distance, err = cg.checkCollision(xGeometry, yGeometry); err != nil {
				return nil, err
			}
			cg.setDistance(xName, yName, distance)
			if !reportDistances && distance <= spatial.CollisionBuffer {
				// collision found, can return early
				return cg, nil
			}
		}
	}
	return cg, nil
}

// checkCollision takes a pair of geometries and returns the distance between them.
// If this number is less than the CollisionBuffer they can be considered to be in collision.
func (cg *collisionGraph) checkCollision(x, y spatial.Geometry) (float64, error) {
	if cg.reportDistances {
		return x.DistanceFrom(y)
	}
	col, err := x.CollidesWith(y)
	if col {
		return math.Inf(-1), err
	}
	return math.Inf(1), err
}

// collisionBetween returns a bool describing if the collisionGraph has a collision between the two entities that are specified by name.
func (cg *collisionGraph) collisionBetween(name1, name2 string) bool {
	if distance, ok := cg.getDistance(name1, name2); ok {
		return distance <= spatial.CollisionBuffer
	}
	return false
}

// collisions returns a list of all the collisions present in the collisionGraph.
func (cg *collisionGraph) collisions() []Collision {
	var collisions []Collision
	for xName, row := range cg.distances {
		for yName, distance := range row {
			if distance <= spatial.CollisionBuffer {
				collisions = append(collisions, Collision{xName, yName, distance})
				if !cg.reportDistances {
					// collision found, can return early
					return collisions
				}
			}
		}
	}
	return collisions
}

// ignoreCollision finds the specified collision and marks it as something never to check for or report.
func (cg *collisionGraph) addCollisionSpecification(specification *Collision) {
	cg.setDistance(specification.name1, specification.name2, math.NaN())
}
