from cogphysics.core import cpObject
from cogphysics.core.physics import OdePhysics as Physics


class TowerScene(object):

    def __init__(self, cpopath):
        self.scene = None
        self.tower = None
        self.blocks = []
        self.table = None

        self.cpopath = cpopath

    @classmethod
    def create(cls, stimname, table=None, cpopath=None):

        # create the object to hold the scene
        obj = cls(cpopath)

        # update the cpopath, if necessary
        if cpopath:
            obj.cpopath = cpopath

        # create an empty scene
        obj.scene = cpObject()

        # load the tower from disk
        try:
            obj.tower = cpObject.loadFile(obj.cpopath, stimname)
        except IOError:
            cponame, strparams = stimname.split("~")
            obj.tower = cpObject.loadFile(obj.cpopath, cponame)

        # set the label (name) for the scene based on the tower's name
        obj.scene.label = stimname
        # give the tower a generic name
        obj.tower.label = 'tower'
        # set the parent of the tower to be the scene
        obj.tower.parent = obj.scene
        # get the list of blocks
        obj.blocks = [x for x in obj.tower.children]

        # create a table, if necessary
        if table is None:
            obj.table = cls.makeTable()
        else:
            obj.table = table

        # set the table's parent to be the scene
        obj.table.parent = obj.scene

        # set block properties
        obj.setBlockProperties()

        # reset initial state
        obj.scene.resetInit(fchildren=True)

        return obj

    @staticmethod
    def makeTable():
        table = cpObject(
            label='table',
            surface='old_cpo_surface',
            model='wood_floor',
            scale=(10, 10, 1),
            posl=(0.0, 0.0, -0.5),
            quatl=(1.0, 0.0, 0.0, 0.0),
            density=None,
            shape='cylinderZ',
            color=(0.45, 0.3, 0.1, 1.0))

        return table

    def setBlockProperties(self):
        pass

    def setGraphics(self):
        pass

    def createPhysics(self):
        self.scene.physics = Physics
        self.scene.enablePhysics()
        self.scene.propagate('physics')

    def destroyPhysics(self):
        if self.scene:
            self.scene.physics = None
            self.scene.propagate('physics')

    def destroy(self):
        self.destroyPhysics()
        for block in self.blocks:
            block.destroy()
        self.blocks = []
        if self.tower:
            self.tower.destroy()
            self.tower = None
        if self.scene:
            self.scene.destroy()
            self.scene = None
