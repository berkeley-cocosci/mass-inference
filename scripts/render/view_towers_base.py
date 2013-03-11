import cogphysics
import cogphysics.lib.physutil as physutil

from cogphysics.core.graphics import PandaGraphics as Graphics
from cogphysics.core.physics import OdePhysics as Physics

import pandac.PandaModules as pm
import libpanda as lp
import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase
p3d.loadPrcFileData("", "prefer-parasite-buffer #f")

import datetime
import os
import sys
import numpy as np


class ViewTowers(ShowBase, object):

    towerscene_type = None

    def __init__(self, cpopath):

        ShowBase.__init__(self)

        # set the path to stimuli objects
        self.cpopath = cpopath

        # position camera, create lights, create the environment (e.g. sky),
        # other misc graphics settings
        self.makeWorld()

        # create a table for the scene (so we don't have to recreate it each
        # time)
        self.table = self.towerscene_type.makeTable()

        # create an empty tower scene
        self.towerscene = None
        self.loadScene(None)

        # create the table and give it graphics
        self.table.graphics = Graphics
        self.table.enableGraphics()

        for mat in self.table.graphics.node.findAllMaterials():
            mat.setShininess(0)
            mat.clearSpecular()
            mat.clearAmbient()
            mat.clearDiffuse()

        # set physics properties
        Physics.setGravity((0.0, 0.0, -9.8))

        # physics variables
        self.phys_accum = 0
        self.phys_step = 1.0 / 1000.0

        # camera variables
        self.cam_start = 0
        self.cam_accum = 0.0
        self.total_cam_time = 2.8
        self.cam_range = np.pi / 2.0
        self.rotating.setH(self.cam_start)

        # task flags
        self.fphysics = False
        self.fspincam = False
        self.fwireframe = False
        self.fdebug = False

        # add tasks
        self.taskMgr.add(self.simulatePhysics, "physics")
        self.taskMgr.add(self.spinCamera, "camera")

        # key bindings dictionart
        self.keybindings = {
            'escape': sys.exit,
            'space': self.togglePhysics,
            'r': self.reset,
            's': self.captureImage,
            'p': self.toggleSpinCamera,
            'w': self.toggleWireframe,
            'arrow_right': self.nextScene,
            'arrow_left': self.prevScene
            }

        # actually accept keybindings
        for key in self.keybindings:
            self.accept(key, self.keybindings[key])

    def loadScene(self, scene, cpopath=None):

        if self.towerscene:
            self.table.parent = None
            self.towerscene.destroy()
            self.towerscene = None

        if not scene:
            return

        if cpopath is None:
            cpopath = self.cpopath

        self.towerscene = self.towerscene_type.create(
            scene, table=self.table, cpopath=cpopath)
        self.reset()

    def placeCamera(self):
        # Position the camera
        base.camera.setPos(0, -10, 1.75)
        #base.camera.setPos(0, -20, 5)
        base.camera.lookAt(0, 0, 1.5)
        # lens = lp.PerspectiveLens()
        # base.cam.node().setLens(lens)
        # base.camLens = lens
        #lens = base.camLens
        #lens.setNearFar(1.0, 1000.0)
        #lens.setFov(40)
        self.rotating = render.attachNewNode("rotating")
        base.camera.reparentTo(self.rotating)
        print base.camera.getPos()

    def createLights(self):
        # Create some lights
        self.lights = pm.NodePath('lights')
        self.lights.reparentTo(render)

        angle1 = -155
        angle2 = 180 + angle1
        r1 = r2 = 5
        z = 8
        at = (1, 0, .02)
        color = tuple(np.ones(4)*3)
        slcolor = 1

        p1 = (np.cos(np.radians(angle1))*r1, np.sin(np.radians(angle1))*r1, z)
        p2 = (np.cos(np.radians(angle2))*r2, np.sin(np.radians(angle2))*r2, z)

        # Create point lights
        plight = pm.PointLight('plight1')
        plight.setColor(color)
        plight.setAttenuation(at)
        light = self.lights.attachNewNode(plight)
        light.setPos(p1)
        light.lookAt(0, 0, 0)
        render.setLight(light)

        plight = pm.PointLight('plight2')
        plight.setColor(tuple(np.array(color)-slcolor))
        plight.setAttenuation(at)
        light = self.lights.attachNewNode(plight)
        light.setPos(p2)
        light.lookAt(0, 0, 0)
        render.setLight(light)

        # Create ambient light
        alight = pm.AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1.0))
        alnp = self.lights.attachNewNode(alight)
        render.setLight(alnp)


        # Create spotlight
        slight = pm.Spotlight('slight')
        slight.setScene(render)
        #slight.setShadowCaster(True, 2048, 2048)
        slight.setColor((slcolor, slcolor, slcolor, 1))
        # slight.getLens().setFov(40, 20)
        # slight.getLens().setNearFar(1,50)
        slight.setAttenuation(at)
        slnp = self.lights.attachNewNode(slight)
        slnp.setPos(p2)
        slnp.lookAt(0, 0, 0)
        render.setLight(slnp)

        # if (base.win.getGsg().getSupportsBasicShaders()==0):
        #     print "Video driver reports that shaders are not supported."
        #     sys.exit()
        # if (base.win.getGsg().getSupportsDepthTexture()==0):
        #     print "Video driver reports that depth textures are not supported."
        #     sys.exit()

        # # base.camLens.setNearFar(1.0,10000)
        # # base.camLens.setFov(75)

        # # creating the offscreen buffer
        # winprops = p3d.WindowProperties.size(512,512)
        # props = p3d.FrameBufferProperties()
        # props.setRgbColor(1)
        # props.setAlphaBits(1)
        # props.setDepthBits(1)
        # LBuffer = base.graphicsEngine.makeOutput(
        #     base.pipe, "offscreen buffer", -2,
        #     props, winprops,
        #     p3d.GraphicsPipe.BFRefuseWindow,
        #     base.win.getGsg(), base.win)
    
        # if (LBuffer == None):
        #    print "Video driver cannot create an offscreen buffer."
        #    sys.exit()

        # Ldepthmap = p3d.Texture()
        # LBuffer.addRenderTexture(
        #     Ldepthmap, 
        #     p3d.GraphicsOutput.RTMBindOrCopy, 
        #     p3d.GraphicsOutput.RTPDepthStencil)
        # if (base.win.getGsg().getSupportsShadowFilter()):
        #     Ldepthmap.setMinfilter(p3d.Texture.FTShadow)
        #     Ldepthmap.setMagfilter(p3d.Texture.FTShadow) 

        # slnp = base.makeCamera(LBuffer)
        # slnp.node().setScene(render)
        # slnp.node().getLens().setFov(40)
        # slnp.node().getLens().setNearFar(1,100)

        # # setting up shader
        # ambient = 0.2
        # render.setShaderInput('light',)
        # render.setShaderInput('Ldepthmap', Ldepthmap)
        # render.setShaderInput('ambient', ambient, 0, 0, 1.0)
        # render.setShaderInput('texDisable', 0, 0, 0, 0)
        # render.setShaderInput('scale', 1, 1, 1, 1)
    
        # # Put a shader on the Light camera
        # lci = lp.NodePath(lp.PandaNode("Light Camera Initializer"))
        # lci.setShader(p3d.Shader.load(
        #     os.path.join(cogphysics.SHADER_PATH, 'caster.sha')))
        # slnp.node().setInitialState(lci.getState())
    
        # # Put a shader on the Main camera.
        # # Some video cards have special hardware for shadow maps.
        # # If the card has that, use it.  If not, use a different
        # # shader that does not require hardware support.
        # mci = lp.NodePath(lp.PandaNode("Main Camera Initializer"))
        # if (base.win.getGsg().getSupportsShadowFilter()):
        #     mci.setShader(p3d.Shader.load(
        #         os.path.join(cogphysics.SHADER_PATH, 'shadow.sha')))
        # else:
        #     mci.setShader(p3d.Shader.load(
        #         os.path.join(cogphysics.SHADER_PATH, 'shadow-nosupport.sha')))
        # base.cam.node().setInitialState(mci.getState())

        # render.setShaderInput('push', 0.04, 0.04, 0.04, 0)

        # slnp.setPos(0,-30,25)
        # slnp.lookAt(0,0,0)
        # slnp.node().getLens().setNearFar(10,100)
        # # slnp.setPos(p2)
        # # slnp.lookAt(0, 0, 0)
        # # slnp.reparentTo(self.lights)


    def makeEnvironment(self):
        modelpath = cogphysics.path(cogphysics.BAM_PATH, 'local')
        texpath = cogphysics.path(cogphysics.TEXTURE_PATH, 'local')

        self.environment = loader.loadModel('smiley.egg')
        self.environment.clearMaterial()
        self.environment.clearTexture()
        self.environment.setAttrib(lp.CullFaceAttrib.make(
            lp.CullFaceAttrib.MCullCounterClockwise))
        self.environment.setScale(20, 20, 20)
        self.environment.setPos(0, 0, 10)
        #self.environment.reparentTo(render)
        self.envtex = loader.loadTexture(
            os.path.join(texpath, 'sky_map.png'))
        self.environment.setTexture(self.envtex, 1)

        for mat in self.environment.findAllMaterials():
            mat.clearDiffuse()
            mat.clearAmbient()
            mat.clearSpecular()
            mat.setShininess(0)

        self.environment.setLightOff()
        self.environment.setColor((.6, .6, .6, 1))

    def makeWorld(self):
        self.placeCamera()
        self.createLights()
        #self.makeEnvironment()
        base.win.setClearColor((0.25, 0.25, 0.45, 1.0))
        base.disableMouse()
        render.setShaderAuto()

    def toggleDebug(self):

        if not self.fdebug:
            render.setShaderOff()
        else:
            render.setShaderAuto()

        self.fdebug = not(self.fdebug)

    def toggleSpinCamera(self, task=None):
        self.fspincam = not(self.fspincam)

    def togglePhysics(self):
        self.fphysics = not(self.fphysics)

    def nextScene(self):
        if self.fwireframe:
            self.toggleWireframe()
        self.sidx = min(self.sidx+1, len(self.scenes)-1)
        self.loadScene(self.scenes[self.sidx])
        print self.scenes[self.sidx]

    def prevScene(self):
        if self.fwireframe:
            self.toggleWireframe()
        self.sidx = max(self.sidx-1, 0)
        self.loadScene(self.scenes[self.sidx])
        print self.scenes[self.sidx]

    def reset(self):
        fwireframe = self.fwireframe
        if fwireframe:
            self.toggleWireframe()

        self.fphysics = False
        self.phys_accum = 0.0

        self.towerscene.destroyPhysics()
        Physics.destroyGlobals()
        Physics.createGlobals()
        self.towerscene.createPhysics()
        self.towerscene.scene.reset(fchildren=True)
        self.towerscene.setBlockProperties()
        self.towerscene.setGraphics()

        if fwireframe:
            self.toggleWireframe()

    def _wobberp(self, t, A, profile="sine_sine"):
        """ Compute wobble angle at time 't', where t \in [0,1] and 'A' is the
        maximum angle over t=[0,1].
        
        profile='sine_sine': uses sin(2*pi*t) * sin(pi*t)
        profile='sine': uses sin(2*pi*t)
        """

        if profile == "sine_sine":
            X = np.sin(2. * np.pi * t) * np.sin(np.pi * t)
            mx = 4. / (3 * np.sqrt(3)) # from wolfram alpha
        elif profile == "sine":
            X = np.sin(2. * np.pi * t)
            mx = 1.
            
        # Scale to [-A, A]
        X *= float(A) / mx

        return X

    def spinCamera(self, task):
        if self.fspincam:
            self.cam_accum += globalClock.getDt()
            self.cam_accum = min(self.cam_accum, self.total_cam_time)
            if self.cam_accum == self.total_cam_time:
                self.fspincam = False

            norm_cam_time = self.cam_accum / self.total_cam_time
            ang = norm_cam_time * self.cam_range
            deg = np.degrees(ang)
            self.rotating.setH(deg + self.cam_start)

        return task.cont

    def spinCameraForever(self, task):
        if self.fspincam:
            time = globalClock.getDt()
            deg = time * 60
            self.rotating.setH(self.rotating.getH() + deg)
        return task.cont

    def simulatePhysics(self, task):
        if self.fphysics:
            frametime = globalClock.getDt()
            self.phys_accum += frametime
            numsteps = int(self.phys_accum / self.phys_step)
            physutil.step(
                Physics,
                cpos=self.towerscene.blocks,
                numsteps=numsteps,
                physstep=self.phys_step,
                forces=[])
            self.phys_accum -= numsteps * self.phys_step
        return task.cont
    
    def captureImage(self, name=None):
        """
        Captures current screenshot and saves to disk

        name : (optional) filename to write screenshot image to. if
        not provided, defaults to 'screenshot#'
        """

        # if name is not provided, use defaults
        if name is None:

            # set screenshot name
            time = str(datetime.datetime.now())
            time = time.replace(" ", "_")
            name = './screenshot_%s_%s.jpg' % (
                self.towerscene.scene.label, time)

        # Capture the screenshot and save it
        filename = self.screenshot(
            namePrefix=name,
            defaultFilename=False,
            source=self.win)

        if filename is not None:
            # print results
            print "Screenshot saved to: %s" % name
        else:
            # error handling
            raise ValueError("Screenshot NOT saved to filename: '%s'" % name)

